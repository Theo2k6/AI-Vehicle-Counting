import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# ==================== CẤU HÌNH ĐÃ ĐIỀN SẴN ====================
SEQ_DIR       = "data/MOT17/MOT17/train/MOT17-04-FRCNN"
OUTPUT_PRED   = "pred_MOT17-04.txt"
MODEL_PATH    = "yolov8m.pt"
CONF_THRESHOLD= 0.10
IOU_THRESHOLD = 0.5
# ==============================================================

def read_seqinfo(seq_dir):
    ini = os.path.join(seq_dir, "seqinfo.ini")
    info = {}
    with open(ini, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("["):
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return int(info["seqLength"]), int(info["imWidth"]), int(info["imHeight"]), float(info["frameRate"])

def main():
    seq_length, w, h, fps = read_seqinfo(SEQ_DIR)
    img_dir = os.path.join(SEQ_DIR, "img1")
    frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Sequence: {os.path.basename(SEQ_DIR)} | {seq_length} frames")

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()

    with open(OUTPUT_PRED, "w") as fout:
        for fidx, img_path in enumerate(frames, start=1):
            frame = cv2.imread(img_path)
            if frame is None: continue

            frame_640 = cv2.resize(frame, (640, 640))
            # SỬA Ở ĐÂY: imgsz=1280
            results = model(frame_640, imgsz=1280, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            if detections.class_id is not None and len(detections) > 0:
                mask = detections.class_id == 0  # Chỉ giữ người
                detections = detections[mask]

            if len(detections) == 0: continue

            detections = tracker.update_with_detections(detections)
            if detections.tracker_id is None: continue

            sx, sy = w / 640.0, h / 640.0
            for i, tid in enumerate(detections.tracker_id):
                if tid is None: continue
                x1, y1, x2, y2 = detections.xyxy[i]
                x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
                bw, bh = x2 - x1, y2 - y1
                conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                fout.write(f"{fidx},{int(tid)},{x1:.2f},{y1:.2f},{bw:.2f},{bh:.2f},{conf:.4f},-1,-1\n")

            if fidx % 100 == 0:
                print(f"  Frame {fidx}/{seq_length}")
    print(f"\nXong. File lưu tại: {OUTPUT_PRED}")

if __name__ == "__main__":
    main()
