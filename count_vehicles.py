import os
import csv
import cv2
import numpy as np
import argparse  # Thêm thư viện này
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

# ==================== CẤU HÌNH ====================
# Sử dụng argparse để nhận tên file video từ dòng lệnh
parser = argparse.ArgumentParser(description="Đếm xe qua vạch")
parser.add_argument("--video", type=str, default="data/traffic_720p.mp4", help="Đường dẫn tới file video")
args = parser.parse_args()

VIDEO_PATH    = args.video

OUTPUT_VIDEO  = "output/output_counted.mp4"
OUTPUT_CSV    = "output/crossing_log.csv"
OUTPUT_SUMMARY= "output/summary_counts.csv"

MODEL_PATH    = "yolov8m.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5

LINE_START    = (200, 420)
LINE_END      = (440, 420)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
SCALE_TO_ORIGINAL = True
# ==============================================================

def line_side_value(x1, y1, x2, y2, px, py):
    """Tính xem điểm (px, py) nằm ở phía nào của đường thẳng (x1,y1)-(x2,y2)"""
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A * px + B * py + C

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w_orig}x{h_orig} @ {fps:.1f} fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_size = (w_orig, h_orig) if SCALE_TO_ORIGINAL else (640, 640)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, out_size)

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()

    track_side = {}
    crossed_ids = set()
    counts = defaultdict(lambda: {"up": 0, "down": 0})
    crossing_log = []

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_640 = cv2.resize(frame, (640, 640))
        results = model(frame_640, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0:
            mask = np.array([int(c) in VEHICLE_CLASSES for c in detections.class_id])
            detections = detections[mask]

        if len(detections) > 0:
            detections = tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                for i, track_id in enumerate(detections.tracker_id):
                    if track_id is None:
                        continue

                    xyxy = detections.xyxy[i]
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2

                    s = line_side_value(LINE_START[0], LINE_START[1], LINE_END[0], LINE_END[1], cx, cy)
                    cur = 1 if s > 0 else (-1 if s < 0 else 0)

                    cls_id = int(detections.class_id[i])
                    cls_name = VEHICLE_CLASSES.get(cls_id, "unknown")

                    if (track_id in track_side and track_id not in crossed_ids):
                        prev = track_side[track_id]
                        if prev != 0 and cur != 0 and prev != cur:
                            direction = "up" if (prev > 0 and cur < 0) else "down"
                            counts[cls_name][direction] += 1
                            crossed_ids.add(track_id)
                            crossing_log.append({
                                "frame": frame_idx, "timestamp": round(frame_idx / fps, 3),
                                "track_id": int(track_id), "class": cls_name, "direction": direction,
                            })

                    if track_id not in crossed_ids:
                        track_side[track_id] = cur

        if SCALE_TO_ORIGINAL:
            sx, sy = w_orig / 640.0, h_orig / 640.0
            det_xyxy = detections.xyxy.copy()
            det_xyxy[:, [0, 2]] *= sx
            det_xyxy[:, [1, 3]] *= sy
            draw_det = sv.Detections(xyxy=det_xyxy, confidence=detections.confidence, class_id=detections.class_id, tracker_id=detections.tracker_id)
            line_pt1, line_pt2 = (int(LINE_START[0] * sx), int(LINE_START[1] * sy)), (int(LINE_END[0] * sx), int(LINE_END[1] * sy))
            canvas = frame
        else:
            draw_det, line_pt1, line_pt2, canvas = detections, LINE_START, LINE_END, frame_640

        labels = []
        for i in range(len(draw_det)):
            cid = int(draw_det.class_id[i])
            cname = VEHICLE_CLASSES.get(cid, "?")
            tid = draw_det.tracker_id[i] if draw_det.tracker_id is not None else -1
            labels.append(f"{cname} #{tid}")

        canvas = box_annotator.annotate(canvas, draw_det)
        canvas = label_annotator.annotate(canvas, draw_det, labels)

        cv2.line(canvas, line_pt1, line_pt2, (0, 255, 255), 2)

        y = 30
        for cname in ["car", "motorcycle", "bus", "truck"]:
            up, down = counts[cname]["up"], counts[cname]["down"]
            cv2.putText(canvas, f"{cname}: UP={up} DOWN={down}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        out.write(canvas)

        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{total_frames} | Đã đếm: {len(crossed_ids)} xe")

    cap.release()
    out.release()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "timestamp", "track_id", "class", "direction"])
        writer.writeheader()
        writer.writerows(crossing_log)

    with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "up", "down", "total"])
        for cname in ["car", "motorcycle", "bus", "truck"]:
            up, down = counts[cname]["up"], counts[cname]["down"]
            writer.writerow([cname, up, down, up + down])

    print("\n=== HOÀN TẤT ĐẾM XE ===")
    for cname in ["car", "motorcycle", "bus", "truck"]:
        up, down = counts[cname]["up"], counts[cname]["down"]
        print(f"  {cname:12s}: UP={up:4d}  DOWN={down:4d}  TOTAL={up+down:4d}")

if __name__ == "__main__":
    main()
