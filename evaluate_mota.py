import os
import numpy as np
import pandas as pd
import motmetrics as mm

# ==================== CẤU HÌNH ====================
GT_PATH   = "data/MOT17/MOT17/train/MOT17-04-FRCNN/gt/gt.txt"
PRED_PATH = "pred_MOT17-04.txt"
SEQ_LENGTH= 1050
IOU_MATCH = 0.7
# ===================================================

# --- BẢN VÁ LỖI NUMPY 2.0 CHO MOTMETRICS ---
import numpy
if not hasattr(numpy, 'asfarray'):
    numpy.asfarray = lambda a, dtype=float: numpy.asarray(a, dtype=dtype)
# -------------------------------------------

def load_mot_txt(path, is_gt=False):
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    df = pd.read_csv(path, header=None, names=cols)
    if is_gt:
        df = df[(df["class"] == 1) & (df["conf"] == 1)].copy()
        df = df[df["vis"] >= 0.0]
    df = df[(df["frame"] >= 1) & (df["frame"] <= SEQ_LENGTH)]
    return df

def to_xyxy(df):
    return np.column_stack([df["x"].values, df["y"].values, df["x"].values + df["w"].values, df["y"].values + df["h"].values])

def main():
    gt = load_mot_txt(GT_PATH, is_gt=True)
    pred = load_mot_txt(PRED_PATH, is_gt=False)

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in range(1, SEQ_LENGTH + 1):
        g = gt[gt["frame"] == frame]
        p = pred[pred["frame"] == frame]
        gt_ids, pred_ids = g["id"].values, p["id"].values
        if len(g) == 0 and len(p) == 0: continue
        gt_boxes, pred_boxes = to_xyxy(g), to_xyxy(p)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            dist = np.empty((len(gt_boxes), len(pred_boxes)))
        else:
            dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=IOU_MATCH)
        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_frames", "mota", "motp", "num_switches", "num_false_positives", "num_misses", "num_objects", "idf1"], name="MOT17")

    print("\n=== KẾT QUẢ ĐÁNH GIÁ MOTA ===")
    print(summary.to_string())

    mota = float(summary["mota"].values[0])
    print(f"\n=> MOTA = {mota:.4f} -> {'ĐẠT (>=0.50)' if mota >= 0.50 else 'CHƯA ĐẠT (<0.50)'}")

    with open("output/mota_report.txt", "w") as f:
        f.write(summary.to_string())
        f.write(f"\n\nMOTA = {mota:.4f}\n")
        f.write("Kết luận: " + ("ĐẠT" if mota >= 0.50 else "CHƯA ĐẠT") + "\n")

if __name__ == "__main__":
    main()
