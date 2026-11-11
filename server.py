"""
Cloud Run Server API — Seamlessly bridging Unity File Uploads to the 3DGS Pipeline.
"""
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import your native pipeline and config tools
import pipeline
from config import PipelineConfig

app = FastAPI(title="Splat Analyzer API (Cloud Run)")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.post("/analyze")
async def run_analyzer(
    # 1. Core Fix: Must use File(...) to explicitly inform the framework this is a binary file stream
    file: UploadFile = File(...),
    
    # 2. Core Fix: All other attached parameters must explicitly use Form(...)
    prompt: str = Form(...),
    quality: str = Form("high"),
    score_threshold: float = Form(0.05),
    min_votes: int = Form(40),
    min_peak_score: float = Form(0.40),
    
    # Spatial override parameters from Unity
    override_center: bool = Form(False),
    center_x: float = Form(0.0),
    center_y: float = Form(0.0),
    center_z: float = Form(0.0),
    
    override_cam_radius: bool = Form(False),
    cam_radius: float = Form(2.05),
    
    override_scene_radius: bool = Form(False),
    scene_radius: float = Form(6.0),
    
    num_nodes: int = Form(5),
    max_results: int = Form(0)
):
    # Create a unique job directory in Cloud Run's in-memory /tmp storage
    job_id = str(uuid.uuid4())
    job_dir = Path(f"/tmp/jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".ply", ".spz"]:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Only .ply and .spz files are supported.")
        
    scene_path = job_dir / f"scene{suffix}"
    
    try:
        # Save the uploaded binary stream to the local container path
        with open(scene_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[{job_id}] File uploaded successfully: {scene_path}")
        
        # 3. Replicate the override logic from run_local.py
        overrides = {
            "quality": quality,
            "score_threshold": score_threshold,
            "min_votes": min_votes,
            "min_peak_score": min_peak_score,
            "num_nodes": num_nodes,
        }
        
        if max_results > 0:
            overrides["max_results"] = max_results
            
        if override_center:
            overrides.update({
                "manual_center_x": center_x, 
                "manual_center_y": center_y, 
                "manual_center_z": center_z,
                "scan_nodes": [[center_x, center_y, center_z]]
            })
            
        if override_cam_radius:
            overrides["camera_radius"] = cam_radius
            
        if override_scene_radius:
            overrides["scene_radius"] = scene_radius

        # Instantiate the pipeline configuration
        cfg = PipelineConfig.from_overrides(**overrides)
            
        # 4. Execute the master pipeline
        interactions = pipeline.run_pipeline(
            ply_path=str(scene_path),
            prompt=prompt,
            job_dir=str(job_dir),
            cfg=cfg
        )
        
        # 5. Return the JSON payload directly to Unity
        # Keep the deployed v3 HTTP field set; v4 metadata remains in local exports.
        public_fields = ("label", "position", "rotation", "scale", "frames")
        return {"objects": [
            {key: obj[key] for key in public_fields} for obj in interactions
        ]}
        
    except Exception as e:
        import traceback
        error_msg = f"Pipeline Error: {e}\n{traceback.format_exc()}"
        print(f"[{job_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 6. Crucial for Cloud Run: Free up memory container space after completion
        shutil.rmtree(job_dir, ignore_errors=True)
        print(f"[{job_id}] Cleaned up temp directory.")