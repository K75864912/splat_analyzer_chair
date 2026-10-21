"""
Master pipeline:
  1. Render spherical camera views from multiple positions (render_cameras.py)
  2. Run OWLv2 open-vocabulary detection on every frame
  3. Back-project 2D boxes to 3D using per-pixel depth maps
  4. Cluster per-label detections → single 3D bounding box per object
  5. Write interactions.json in WebXR format

Usage:
  python pipeline.py --ply scene.ply --prompt "chair, desk" --job_dir /tmp/job123
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

# Let OWLv2 ops not yet implemented on Apple MPS fall back to CPU instead of hard-erroring.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO

import render_cameras
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY


def _select_device() -> str:
    """Compute device for OWLv2. Order: CUDA → Apple MPS → CPU.
    Set WMD_DEVICE=cuda|mps|cpu to override (e.g. WMD_DEVICE=cpu if MPS misbehaves)."""
    override = os.getenv("WMD_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 2-D → 3-D lifting helpers
# ---------------------------------------------------------------------------

def _unproject_box(box_2d, depth, K_inv, c2w):
    """
    Unproject a 2-D bounding box centre to a 3-D ray and place it at `depth`
    along that ray in world space.
    """
    x1, y1, x2, y2 = box_2d
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0
    w_px = x2 - x1
    h_px = y2 - y1

    p_cam_norm = K_inv @ np.array([cx_px, cy_px, 1.0])
    ray_cam = p_cam_norm / np.linalg.norm(p_cam_norm)
    point_cam = ray_cam * (depth / ray_cam[2])
    point_world = (c2w[:3, :3] @ point_cam) + c2w[:3, 3]

    return point_world, (w_px, h_px)


def _pixel_size_to_world(w_px, h_px, depth, fl_x, fl_y):
    world_w = (w_px / fl_x) * depth
    world_h = (h_px / fl_y) * depth
    return world_w, world_h


def _cluster_detections(detections, eps_m=0.5, max_per_label=3,
                        min_votes=8, min_peak_score=0.35, diagnostics=None):
    """
    Anchor-based greedy clustering with false-positive suppression.

    The seed detection's position is used as a FIXED anchor — no centroid drift.
    Drift was causing two nearby same-label objects (e.g. two sofas) to bleed
    into each other's clusters because the shifting centroid would migrate toward
    the second object and absorb its detections.
    """
    by_label = defaultdict(list)
    for det in detections:
        by_label[det["label"]].append(det)

    results = []
    for label, dets in by_label.items():
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        positions = np.array([d["position"] for d in dets])
        scales    = np.array([d["scale"]    for d in dets])
        scores    = np.array([d["score"]    for d in dets])

        used = [False] * len(dets)
        clusters = []
        for i in range(len(dets)):
            if used[i]:
                continue
            # Fixed anchor — never updated. Prevents cluster from drifting
            # toward a neighbouring object and stealing its detections.
            anchor = positions[i].copy()
            cluster_idx = [i]
            for j in range(i + 1, len(dets)):
                if not used[j] and np.linalg.norm(anchor - positions[j]) < eps_m:
                    cluster_idx.append(j)
            for j in cluster_idx:
                used[j] = True

            raw_vote_count = len(cluster_idx)
            # One highest-confidence representative per camera pose, not per box.
            best_by_view = {}
            for j in cluster_idx:
                view_id = dets[j].get("view_id", dets[j]["frame_idx"])
                if view_id not in best_by_view or scores[j] > scores[best_by_view[view_id]]:
                    best_by_view[view_id] = j
            cluster_idx = list(best_by_view.values())
            peak_score = float(scores[cluster_idx].max())
            vote_count = len(cluster_idx)
            reasons = []
            if vote_count < min_votes:
                reasons.append("insufficient_unique_views")
            if peak_score < min_peak_score:
                reasons.append("low_peak_score")
            if diagnostics is not None:
                diagnostics.append({
                    "label": label, "anchor": anchor.tolist(),
                    "raw_votes": raw_vote_count, "unique_view_votes": vote_count,
                    "peak_score": peak_score, "eps_scene_units": float(eps_m),
                    "passes_thresholds": not reasons, "rejection_reasons": reasons,
                    "representative_frame_indices": [int(dets[j]["frame_idx"]) for j in cluster_idx],
                })
            if reasons:
                continue

            cluster_pos   = (positions[cluster_idx] * scores[cluster_idx, None]).sum(0) / scores[cluster_idx].sum()

            # v3.1 original
            # cluster_scale = np.median(scales[cluster_idx], axis=0)
            
            cluster_scales = scales[cluster_idx]
            volumes = cluster_scales[:, 0] * cluster_scales[:, 1] * cluster_scales[:, 2]
            # rank by volumn and pick the 25th percentile as the cluster scale
            sorted_indices = np.argsort(volumes)
            target_idx = sorted_indices[len(sorted_indices) // 4]
            cluster_scale = cluster_scales[target_idx]

            cluster_scale = np.maximum(cluster_scale, 0.1)
            # Carry the raw member dicts so callers can trace back to source frames
            member_dets = [dets[j] for j in cluster_idx]
            clusters.append((peak_score, vote_count, label, cluster_pos, cluster_scale, member_dets))

        clusters.sort(key=lambda c: c[0], reverse=True) # original: sort by peak score
        #clusters.sort(key=lambda c: c[1], reverse=True) # v3.3: sort by vote count instead of peak score

        for peak, votes, lbl, pos, scale, members in clusters[:max_per_label]:
            print(f"  [cluster] {lbl}: {votes} votes, peak={peak:.2f}, pos={pos.round(2)}")
            #results.append((lbl, pos, scale, members))
            results.append((lbl, pos, scale, members, votes))  # v3.3: keep all clusters for max_results filtering later

    return results


# ---------------------------------------------------------------------------
# YOLOE-26 detection
# ---------------------------------------------------------------------------

def _mask_world_position(mask, depth, K_inv, c2w, min_pixels=25):
    """Coordinatewise median of surface points, NOT a chair's physical center."""
    valid = mask & np.isfinite(depth) & (depth > .01)
    y,x = np.nonzero(valid)
    if len(x)<min_pixels: return None, {'valid_pixels':len(x)}
    z = depth[y,x].astype(np.float64)
    # Use every pixel (no stochastic sampling); pixel convention matches pipeline.
    rays = np.column_stack([x,y,np.ones(len(x))]) @ K_inv.T
    cam = rays * (z/rays[:,2])[:,None]
    world = cam @ c2w[:3,:3].T + c2w[:3,3]
    pos = np.median(world,axis=0)
    q = np.quantile(z,[.1,.5,.9])
    return pos, dict(valid_pixels=len(x),depth_p10=float(q[0]),depth_median=float(q[1]),depth_p90=float(q[2]))

def _explicit_nms(boxes, scores, classes, threshold=0.60):
    order = np.argsort(-scores, kind='stable')
    keep, removed = [], []
    while len(order):
        i = int(order[0]); keep.append(i); rest = order[1:]
        if not len(rest): break
        wh = np.maximum(0, np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2]))
        inter = wh.prod(axis=1)
        areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-12)
        suppress = (classes[rest] == classes[i]) & (iou > threshold)
        for j, v in zip(rest[suppress], iou[suppress]):
            removed.append({'removed_index': int(j), 'kept_index': i, 'iou': float(v)})
        order = rest[~suppress]
    return keep, removed


def _run_yoloe_26(frames_dir: Path, labels: list[str], transforms: dict, scene_radius: float,
                    score_threshold: float = 0.12):
    """
    Run YOLOE-26 on all rendered frames.
    """
    device = _select_device()
    print(f"[pipeline] Loading YOLOE-26 on {device} …")
    
    # substitute the path to your YOLOE-26 model weights here 
    # you can substitute "l" in the filename for large, "m" for medium, "s" for small, "x" for extra-large, "n" for nano
    model = YOLO("yoloe-26x-seg.pt")
    
    # we can set open-vocabulary classes by passing a list of labels to model.set_classes()
    model.set_classes(labels)

    fl_x = transforms["fl_x"]
    fl_y = transforms["fl_y"]
    cx   = transforms["cx"]
    cy   = transforms["cy"]
    W    = transforms["w"]
    H    = transforms["h"]

    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    raw_detections = []
    position_mode = os.getenv("SPLAT_POSITION_MODE", "mask").lower()
    if position_mode not in {"mask", "center"}:
        raise ValueError("SPLAT_POSITION_MODE must be mask or center")
    position_audit = {"mode": position_mode, "mask_used": 0,
                      "fallback_reasons": {}, "total_detections": 0}
    post_nms_iou = float(os.getenv("SPLAT_POST_NMS_IOU", "0.60"))
    if not 0.0 <= post_nms_iou <= 1.0:
        raise ValueError("SPLAT_POST_NMS_IOU must be between 0 and 1; 1 disables suppression")
    nms_audit = []
    print(f"[detect] Explicit per-class post-NMS IoU={post_nms_iou}")

    scene_center  = np.array(transforms.get("scene_center", [0.0, 0.0, 0.0]))
    stored_radius = transforms.get("scene_radius", scene_radius)

    # 1. 预先收集所有有效图片的路径和元数据
    valid_frames = []
    view_ids = {}
    frame_view_ids = {}
    for frame_idx, frame_meta in enumerate(transforms["frames"]):
        pose_key = tuple(np.round(np.asarray(frame_meta["transform_matrix"]), 6).ravel())
        frame_view_ids[frame_idx] = view_ids.setdefault(pose_key, len(view_ids))
        frame_path = frames_dir.parent / frame_meta["file_path"]
        if frame_path.exists():
            valid_frames.append((frame_idx, frame_meta, str(frame_path)))

    # 4070 Super 拥有 12GB 显存，设为 32 或 64 是最安全的甜点区间
    YOLO_BATCH_SIZE = 32
    print(f"  [detect] Running YOLO chunked inference (batch_size={YOLO_BATCH_SIZE})...")

    # 2. 手动分块 (Chunking)，强制切断 YOLO 一次性吞噬 960 张图的企图
    for i in range(0, len(valid_frames), YOLO_BATCH_SIZE):
        chunk_frames = valid_frames[i : i + YOLO_BATCH_SIZE]
        chunk_paths = [item[2] for item in chunk_frames]

        # Chunk size limits inputs per call; VRAM depends on model and images.
        results = model.predict(
            source=chunk_paths, 
            conf=score_threshold,           
            iou=0.85,            
            imgsz=max(W, H),     
            device=device,
            verbose=False,
            stream=True
        )

        # 3. 顺序遍历当前块的结果
        for batch_idx, res in enumerate(results):
            # 👑 核心优化 1：先看有没有检测到东西
            boxes = res.boxes.xyxy.cpu().numpy()
            
            # 如果当前帧空空如也，直接跳过！不读硬盘，瞬间完成！
            if len(boxes) == 0:
                continue

            scores = res.boxes.conf.cpu().numpy()
            class_ids = res.boxes.cls.cpu().numpy().astype(int)
            before = len(boxes)
            keep, removed = _explicit_nms(boxes, scores, class_ids, post_nms_iou)
            nms_audit.append({
                "frame_idx": int(chunk_frames[batch_idx][0]), "before": before,
                "after": len(keep), "removed": removed,
            })
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

            frame_idx, frame_meta, frame_path_str = chunk_frames[batch_idx]
            
            c2w = np.array(frame_meta["transform_matrix"], dtype=np.float64)
            cam_pos = c2w[:3, 3]

            scan_nodes = transforms.get("scan_nodes", [])
            if scan_nodes and len(scan_nodes) > 0:
                dists = [np.linalg.norm(cam_pos - np.array(n)) for n in scan_nodes]
                cam_to_center = min(dists)
            else:
                cam_to_center = np.linalg.norm(cam_pos - scene_center)
                
            fallback_depth = cam_to_center if cam_to_center > stored_radius * 0.3 else stored_radius

            # 👑 核心优化 2：只有在确定有物体时，才去硬盘读取笨重的 .npy 深度图
            depth_npy_path = (frames_dir.parent / frame_meta.get("depth_path", "").replace(".png", ".npy"))
            depth_map = None
            if depth_npy_path.exists():
                depth_map = np.load(str(depth_npy_path)).astype(np.float64)

            selected_masks = None
            mask_reason = None
            if position_mode == "mask":
                if depth_map is None:
                    mask_reason = "missing_depth"
                elif res.masks is None:
                    mask_reason = "missing_masks"
                elif len(res.masks.data) != before:
                    mask_reason = "mask_count_mismatch"
                elif tuple(res.masks.data.shape[1:]) != tuple(depth_map.shape):
                    mask_reason = "mask_shape_mismatch"
                else:
                    # Preserve mask holes and align masks using the same NMS indices.
                    selected_masks = res.masks.data[keep].cpu().numpy() > 0.5

            # Keep old size estimates; change only the representative position.
            for det_idx, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                label = labels[cls_id] 
                
                bx1, by1, bx2, by2 = box
                cx_px = int(np.clip((bx1 + bx2) / 2, 0, W - 1))
                cy_px = int(np.clip((by1 + by2) / 2, 0, H - 1))

                if depth_map is not None:
                    h5, w5 = depth_map.shape
                    py0 = max(0, cy_px - 2); py1 = min(h5, cy_px + 3)
                    px0 = max(0, cx_px - 2); px1 = min(w5, cx_px + 3)
                    patch = depth_map[py0:py1, px0:px1].ravel()
                    valid = patch[patch > 0.01]
                    sampled = float(np.median(valid)) if valid.size > 0 else 0.0
                    box_depth = sampled if sampled > 0.01 else fallback_depth
                else:
                    box_depth = fallback_depth

                world_pos, (w_px, h_px) = _unproject_box(box, box_depth, K_inv, c2w)
                world_w, world_h = _pixel_size_to_world(w_px, h_px, box_depth, fl_x, fl_y)
                world_d = (world_w + world_h) / 2.0

                if position_mode == "mask":
                    reason = mask_reason
                    if selected_masks is not None:
                        mask_pos, _ = _mask_world_position(
                            selected_masks[det_idx], depth_map, K_inv, c2w)
                        if mask_pos is not None:
                            world_pos = mask_pos
                            position_audit["mask_used"] += 1
                        else:
                            reason = "fewer_than_25_valid_pixels"
                    if reason is not None:
                        reasons = position_audit["fallback_reasons"]
                        reasons[reason] = reasons.get(reason, 0) + 1
                position_audit["total_detections"] += 1

                raw_detections.append({
                    "label":     label,
                    "score":     float(score),
                    "position":  world_pos,
                    "scale":     np.array([world_w, world_h, world_d]),
                    "frame_idx": frame_idx,
                    "view_id": frame_view_ids[frame_idx],
                    "box_2d":    [float(bx1), float(by1), float(bx2), float(by2)],
                })
                print(f"  [detect] {label} ({score:.2f}) depth={box_depth:.2f} "
                      f"@ {world_pos.round(3)} {os.path.basename(frame_path_str)}")

    with open(frames_dir.parent / "position_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(position_audit, f, indent=2)
    print(f"[position-audit] {position_audit}")
    before_total = sum(row["before"] for row in nms_audit)
    after_total = sum(row["after"] for row in nms_audit)
    with open(frames_dir.parent / "nms_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({"post_nms_iou": post_nms_iou,
                   "model_head_end2end": bool(getattr(model.model.model[-1], "end2end", False)),
                   "frames_processed": len(valid_frames),
                   "frames_with_boxes": len(nms_audit),
                   "before_total": before_total, "after_total": after_total,
                   "removed_total": before_total - after_total,
                   "frames": nms_audit}, f, indent=2)
    print(f"[nms-audit] before={before_total}, after={after_total}, "
          f"removed={before_total - after_total}")
    return raw_detections

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_numbered_views(job_dir, interactions, frame_annotations, marker_size):
    """Viewer-compatible numbered labels; never mutate production categories."""
    numbered = []
    for obj in interactions:
        item = dict(obj)
        if "display_label" in item:
            item["category"] = item["label"]
            item["label"] = item["display_label"]
        numbered.append(item)
    annotations = {
        key: [dict(a, label=numbered[a["object_idx"]]["label"]) for a in values]
        for key, values in frame_annotations.items()
    }
    payload = {"objects": numbered, "frame_annotations": annotations}
    with open(Path(job_dir) / "interactions_numbered.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    centers = []
    for obj in numbered:
        item = dict(obj)
        if "object_id" in item:
            item["scale"] = {axis: float(marker_size) for axis in "xyz"}
        centers.append(item)
    with open(Path(job_dir) / "interactions_centers.json", "w", encoding="utf-8") as f:
        json.dump({"objects": centers, "frame_annotations": annotations}, f, indent=2)


def run_pipeline(ply_path: str, prompt: str, job_dir: str,
                 cfg: PipelineConfig | None = None) -> list[dict]:
    cfg = cfg or PipelineConfig()
    cluster_eps_frac = float(os.getenv("SPLAT_CLUSTER_EPS_FRAC", "0.09"))
    if not math.isfinite(cluster_eps_frac) or cluster_eps_frac <= 0:
        raise ValueError("SPLAT_CLUSTER_EPS_FRAC must be finite and positive")
    print(f"[pipeline] position={os.getenv('SPLAT_POSITION_MODE', 'mask')}, "
          f"cluster_eps_frac={cluster_eps_frac}")
    job_dir = Path(job_dir)

    print("[pipeline] Step 1: Rendering camera views …")
    transforms_path = render_cameras.render_views(ply_path, str(job_dir), cfg)

    with open(transforms_path) as f:
        transforms = json.load(f)

    frames_dir = job_dir / "frames"

    cam_positions = np.array([
        frame["transform_matrix"]
        for frame in transforms["frames"]
    ])[:, :3, 3]
    scene_radius = float(np.linalg.norm(cam_positions, axis=1).mean())

    labels = [l.strip() for l in prompt.split(",") if l.strip()]
    if not labels:
        raise ValueError("prompt must contain at least one label")

    print(f"[pipeline] Step 2: Detecting {labels} in {len(transforms['frames'])} frames using YOLOE-26 …")
    raw_detections = _run_yoloe_26(frames_dir, labels, transforms, scene_radius,
                                     score_threshold=cfg.score_threshold)

    frame_annotations: dict = {}   # frame_idx (str) → [{label, object_idx, box, score}]

    cluster_diagnostics = []
    if not raw_detections:
        print("[pipeline] No detections above threshold.")
        interactions = []
    else:
        print(f"[pipeline] Step 3: Clustering {len(raw_detections)} raw detections …")
        clustered = _cluster_detections(
            raw_detections,
            eps_m=transforms.get("scene_radius", scene_radius) * cluster_eps_frac,
            #eps_m=0.6,  # 0.6 meter is a reasonable default for indoor chairs
            #max_per_label=3,
            max_per_label=200,
            min_votes=cfg.min_votes,
            min_peak_score=cfg.min_peak_score,
            diagnostics=cluster_diagnostics,
        )

        # === v3.3: filter the top-k results if max_results is specified ===
        if getattr(cfg, 'max_results', None) is not None:
            print(f"[pipeline] Truncating top {cfg.max_results} results globally based on peak score.")
            clustered = clustered[:cfg.max_results]

        interactions = []
        for obj_idx, (label, pos, scale, members, votes) in enumerate(clustered):
            # Deduplicate members by frame_idx (keep highest-score per frame)
            best: dict = {}
            for m in members:
                fi = m["frame_idx"]
                if fi not in best or m["score"] > best[fi]["score"]:
                    best[fi] = m

            obj_frames = []
            for m in sorted(best.values(), key=lambda x: x["score"], reverse=True):
                fi = m["frame_idx"]
                obj_frames.append({
                    "frame_idx": fi,
                    "box":       m["box_2d"],
                    "score":     round(m["score"], 4),
                })
                fkey = str(fi)
                frame_annotations.setdefault(fkey, []).append({
                    "label":      label,
                    "object_idx": obj_idx,
                    "box":        m["box_2d"],
                    "score":      round(m["score"], 4),
                })

            interactions.append({
                "label":    label,
                "object_id": f"obj_{obj_idx:03d}",
                "display_label": f"{label}_{obj_idx:03d}",
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "scale":    {"x": float(scale[0]), "y": float(scale[1]), "z": float(scale[2])},
                "frames":   obj_frames,
            })

    # Includes rejected candidates; threshold pass does not guarantee export
    # because max_per_label/max_results can still limit output.
    diagnostic_payload = {
        "position_mode": os.getenv("SPLAT_POSITION_MODE", "mask"),
        "cluster_eps_frac": cluster_eps_frac,
        "min_votes": cfg.min_votes, "min_peak_score": cfg.min_peak_score,
        "vote_unit": "unique_camera_pose_rounded_6_decimals",
        "raw_detection_count": len(raw_detections),
        "exported_object_count": len(interactions),
        "candidates": cluster_diagnostics,
    }
    with open(job_dir / "cluster_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)
    print(f"[cluster-audit] candidates={len(cluster_diagnostics)}, "
          f"threshold_pass={sum(c['passes_thresholds'] for c in cluster_diagnostics)}, "
          f"exported={len(interactions)}")

    # === v3: add multiple scan nodes to the output for reference ===
    scan_nodes = transforms.get("scan_nodes", [])
    if not scan_nodes:
        scan_nodes = [transforms.get("scene_center", [0.0, 0.0, 0.0])]
        
    for idx, node in enumerate(scan_nodes):
        center_marker = {
            "label": f"SCAN_NODE_MARKER_{idx}",
            "position": {
                "x": float(node[0]),
                "y": float(node[1]),
                "z": float(node[2])
            },
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "scale": {"x": 0.5, "y": 0.5, "z": 0.5},
            "frames": []
        }
        interactions.append(center_marker)
    # === #

    output_path = job_dir / "interactions.json"
    with open(output_path, "w") as f:
        json.dump({"objects": interactions, "frame_annotations": frame_annotations}, f, indent=2)

    _write_numbered_views(
        job_dir, interactions, frame_annotations,
        transforms.get("scene_radius", scene_radius) * 0.015)
    print("[pipeline] Viewer exports: interactions_numbered.json, interactions_centers.json")
    print(f"[pipeline] Done → {output_path} ({len(interactions)-len(scan_nodes)} objects)")
    return interactions


if __name__ == "__main__":
    _d = PipelineConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",              required=True)
    parser.add_argument("--prompt",           required=True)
    parser.add_argument("--job_dir",          required=True)
    parser.add_argument("--quality",          choices=list(QUALITY_PRESETS.keys()),
                        default=DEFAULT_QUALITY,
                        help="Camera-coverage preset (controls number of views)")
    parser.add_argument("--score_threshold",  type=float, default=_d.score_threshold)
    parser.add_argument("--min_votes",        type=int,   default=_d.min_votes)
    parser.add_argument("--min_peak_score",   type=float, default=_d.min_peak_score)
    args = parser.parse_args()
    cfg = PipelineConfig.from_overrides(
        quality=args.quality,
        score_threshold=args.score_threshold,
        min_votes=args.min_votes,
        min_peak_score=args.min_peak_score,
    )
    result = run_pipeline(args.ply, args.prompt, args.job_dir, cfg)
    print(json.dumps(result, indent=2))
