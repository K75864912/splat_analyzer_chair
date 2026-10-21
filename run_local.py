"""
Local one-command runner for the WorldModelData pipeline.

Runs the *exact same* rendering + detection pipeline the server uses (no
duplicated logic — this is a thin front door over pipeline.run_pipeline) on a
machine with an NVIDIA CUDA GPU.

Example:
  python run_local.py --ply scene.ply --prompt "chair, table" --quality high

Outputs <job_dir>/interactions.json plus the rendered frames/ and transforms.json.
"""

import argparse
import sys
from pathlib import Path

import torch

import pipeline
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY


def main():
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Run 3DGS object detection locally.")
    p.add_argument("--ply",     required=True, help="Path to a .ply or .spz Gaussian Splat file")
    p.add_argument("--prompt",  required=True, help='Comma-separated labels, e.g. "chair, table"')
    p.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()), default=DEFAULT_QUALITY,
                   help="Camera-coverage preset (controls number of views)")
    p.add_argument("--job_dir", default=None, help="Output directory (default: ./out_<name>)")
    p.add_argument("--score_threshold", type=float, default=d.score_threshold)
    p.add_argument("--min_votes",       type=int,   default=d.min_votes)
    p.add_argument("--min_peak_score",  type=float, default=d.min_peak_score)
    # === v3: new manual control parameters ===
    p.add_argument("--center", type=float, nargs=3, default=None, help="Manual center: X Y Z (Legacy)")
    p.add_argument("--node", action="append", nargs=3, type=float, help="Manual scan node: X Y Z (can be used multiple times)")
    # --- v3.1: New arguments for auto-generation ---
    p.add_argument("--num_nodes", type=int, default=1, help="Auto-generate N nodes around the center")
    p.add_argument("--node_spread", type=float, default=0.4, help="Spread radius as a fraction of total room radius")

    p.add_argument("--cam_radius", type=float, default=None, help="Manual camera radius")
    p.add_argument("--scene_radius", type=float, default=None, help="Room physical radius")

    # ---- v3.3: max recognition results ---
    p.add_argument("--max_results", type=int, default=None, help="Max number of top-voted objects to output (optional)")

    args = p.parse_args()

    if not Path(args.ply).exists():
        sys.exit(f"error: file not found: {args.ply}")

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — running on CPU will be extremely slow.\n"
              "         Install a CUDA-matched build of torch + gsplat for usable speed.",
              file=sys.stderr)

    job_dir = args.job_dir or f"out_{Path(args.ply).stem}"
    # ===== original codes =====
    # cfg = PipelineConfig.from_overrides(
    #     quality=args.quality,
    #     score_threshold=args.score_threshold,
    #     min_votes=args.min_votes,
    #     min_peak_score=args.min_peak_score,
    #     bbox_pct_lo=args.bbox_pct_lo,    # new lower bound
    #     bbox_pct_hi=args.bbox_pct_hi,    # new higher bound
    # )
    # ===============================

    # ===== v3.3 overrides =====
    overrides = {
        "quality": args.quality,
        "score_threshold": args.score_threshold,
        "min_votes": args.min_votes,
        "min_peak_score": args.min_peak_score,
        "num_nodes": args.num_nodes,    # v3.1: auto-generate N nodes around the center
        "node_spread_frac": args.node_spread, # v3.1: spread radius as a fraction of total room radius
        "max_results": args.max_results, # v3.3: max number of top-voted objects to output
    }
    
    # if --center X Y Z is provided, inject manual center coordinates into the overrides
    if args.center:
        overrides.update({
            "manual_center_x": args.center[0], 
            "manual_center_y": args.center[1], 
            "manual_center_z": args.center[2]
        })

    # ===== v3: Multi-Node Spatial Scanning overrides =====
    # support both --node X Y Z (multiple) and --center X Y Z (legacy single node)
    if args.node:
        overrides.update({"scan_nodes": args.node})
    elif args.center:
        # if --center is provided, inject it as a single scan node into the overrides
        overrides.update({"scan_nodes": [args.center]})
        
    # if --radius is provided, inject manual radius into the overrides
    #if args.radius:
        #overrides.update({"manual_radius": args.radius})
    if args.cam_radius:
        overrides.update({"camera_radius": args.cam_radius})
    if args.scene_radius:
        overrides.update({"scene_radius": args.scene_radius})

    cfg = PipelineConfig.from_overrides(**overrides)
    # ====================================================

    objects = pipeline.run_pipeline(args.ply, args.prompt, job_dir, cfg)

    out = Path(job_dir)
    print(f"\nDone — {len(objects) - args.num_nodes} object(s) detected.")
    print(f"  results: {out / 'interactions.json'}")
    print(f"  frames:  {out / 'frames'}")


if __name__ == "__main__":
    main()
