import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import shutil
import os
import cv2
import glob
import time
import json
import numpy as np

from src.detectors.detector_factory import DetectorFactory
from src.depth_estimators.depth_factory import DepthEstimatorFactory
from src.fusion.camera_params import CameraIntrinsics
from src.fusion.bbox3d_builder import BBox3DBuilder
from src.visualization.visualizer import Visualizer

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("data_store", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/output", StaticFiles(directory="output"), name="output")

templates = Jinja2Templates(directory="templates")

print("Loading models (this might take a few seconds)...")
detector = DetectorFactory.create_detector("yolo", "weights/yolo11s_openvino_model")
depth_estimator = DepthEstimatorFactory.create_estimator("depth_anything_ov", "weights/depth_anything")
camera_intrinsics = CameraIntrinsics.create_default(640, 480)
bbox3d_builder = BBox3DBuilder(camera_intrinsics)
vis = Visualizer()
print("Models loaded successfully.")

print("Starting OpenVINO warm-up (to prevent slow first inference)...")
dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
detector.detect(dummy_img, conf=0.25, iou=0.45)
depth_estimator.estimate(dummy_img)
print("Warm-up complete. Models are ready!")


def run_image_inference(in_path, out_path, conf_threshold, iou_threshold):
    image = cv2.imread(in_path)
    detections = detector.detect(image, conf=conf_threshold, iou=iou_threshold)
    depth_map = depth_estimator.estimate(image)

    bbox_3d_info = []
    contacts = []
    for i, (box, det_conf, cls_id) in enumerate(detections):
        info = bbox3d_builder.build_from_2d_and_depth(box, depth_map)
        bbox_3d_info.append(info)
        if info:
            cx, cy, _ = info["center_3d"]
            w, h, _ = info["size_3d"]
            contacts.append({
                "track": f"T-{i+1:02d}",
                "status": "CONFIRMED",
                "position": f"{int(cx)}, {int(cy)}",
                "size": f"{int(w)}x{int(h)}",
                "distance": f"{info['distance']:.2f}m",
                "class": int(cls_id),
                "conf": f"{det_conf:.2f}"
            })

    depth_color = vis.colorize_depth(depth_map)
    depth_color = cv2.resize(depth_color, (image.shape[1], image.shape[0]))
    result_img = vis.draw_results(depth_color, detections, bbox_3d_info, depth_map)
    cv2.imwrite(out_path, result_img)
    return contacts


def run_video_first_frame(in_path, conf_threshold, iou_threshold):
    contacts = []
    cap = cv2.VideoCapture(in_path)
    ret, image = cap.read()
    if ret:
        detections = detector.detect(image, conf=conf_threshold, iou=iou_threshold)
        depth_map = depth_estimator.estimate(image)
        for i, (box, det_conf, cls_id) in enumerate(detections):
            info = bbox3d_builder.build_from_2d_and_depth(box, depth_map)
            if info:
                cx, cy, _ = info["center_3d"]
                w, h, _ = info["size_3d"]
                contacts.append({
                    "track": f"T-{i+1:02d}",
                    "status": "CONFIRMED",
                    "position": f"{int(cx)}, {int(cy)}",
                    "size": f"{int(w)}x{int(h)}",
                    "distance": f"{info['distance']:.2f}m",
                    "class": int(cls_id),
                    "conf": f"{det_conf:.2f}"
                })
    cap.release()
    return contacts


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/api/history")
async def get_history():
    history_file = "data_store/history.json"
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(entry):
    history_file = "data_store/history.json"
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

@app.post("/api/upload_infer")
async def upload_infer(file: UploadFile = File(...), conf: float = Form(0.25), iou: float = Form(0.45)):
    file_id = str(int(time.time()))
    ext = os.path.splitext(file.filename)[1].lower()
    in_path = f"uploads/{file_id}{ext}"
    out_path = f"output/{file_id}_result{ext}"

    with open(in_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_video = ext in [".mp4", ".avi", ".mov"]

    if not is_video:
        contacts = run_image_inference(in_path, out_path, conf, iou)
    else:
        contacts = run_video_first_frame(in_path, conf, iou)
        from src.pipeline.pipelines import VideoPipeline
        pipeline = VideoPipeline(detector, depth_estimator, bbox3d_builder, vis)
        out_path = f"output/{file_id}_result.mp4"
        pipeline.run(in_path, out_path)

        import subprocess
        tmp_path = out_path + ".temp.mp4"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", out_path, "-vcodec", "libx264", tmp_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import shutil as sh
            sh.move(tmp_path, out_path)
        except Exception as e:
            print(f"Video conversion failed: {e}")

    entry = {
        "id": file_id,
        "filename": file.filename,
        "type": "video" if is_video else "image",
        "original_url": f"/{in_path}",
        "inferred_url": f"/{out_path}",
        "contacts": contacts
    }
    save_history(entry)
    return entry

@app.post("/api/reinfer")
async def reinfer(original_url: str = Form(...), conf: float = Form(0.25), iou: float = Form(0.45)):
    in_path = original_url.lstrip("/")
    if not os.path.exists(in_path):
        return {"error": "File not found"}

    ext = os.path.splitext(in_path)[1].lower()
    is_video = ext in [".mp4", ".avi", ".mov"]
    file_id = str(int(time.time()))

    if not is_video:
        out_path = f"output/{file_id}_result{ext}"
        contacts = run_image_inference(in_path, out_path, conf, iou)
    else:
        contacts = run_video_first_frame(in_path, conf, iou)
        out_path = f"output/{file_id}_result.mp4"
        from src.pipeline.pipelines import VideoPipeline
        pipeline = VideoPipeline(detector, depth_estimator, bbox3d_builder, vis)
        pipeline.run(in_path, out_path)

        import subprocess
        tmp_path = out_path + ".temp.mp4"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", out_path, "-vcodec", "libx264", tmp_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import shutil as sh
            sh.move(tmp_path, out_path)
        except Exception as e:
            print(f"Video conversion failed: {e}")

    filename = os.path.basename(in_path)
    entry = {
        "id": file_id,
        "filename": filename,
        "type": "video" if is_video else "image",
        "original_url": f"/{in_path}",
        "inferred_url": f"/{out_path}",
        "contacts": contacts
    }
    save_history(entry)
    return entry

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
