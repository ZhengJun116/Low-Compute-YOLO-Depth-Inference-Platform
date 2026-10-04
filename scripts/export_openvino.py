import os
from ultralytics import YOLO
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import openvino as ov


def export_yolo():
    os.makedirs("weights", exist_ok=True)
    model = YOLO("yolo11s.pt")
    model.export(format="openvino", imgsz=640)

    import shutil
    if os.path.exists("weights/yolo11s_openvino_model"):
        shutil.rmtree("weights/yolo11s_openvino_model")
    if os.path.exists("yolo11s_openvino_model"):
        shutil.move("yolo11s_openvino_model", "weights/")


def export_depth_anything():
    os.makedirs("weights/depth_anything", exist_ok=True)

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
    except Exception as e:
        model_id = "LiheYoung/depth-anything-small-hf"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)

    model.eval()

    dummy_input = torch.randn(1, 3, 518, 518)

    ov_model = ov.convert_model(model, example_input=dummy_input)

    ov.save_model(ov_model, "weights/depth_anything/depth_anything.xml")

    processor.save_pretrained("weights/depth_anything")


if __name__ == "__main__":
    export_yolo()
    export_depth_anything()
