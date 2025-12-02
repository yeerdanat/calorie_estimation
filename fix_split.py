import os
import glob
import csv

ROOT = "./Food_Portion_Benchmark/FPB_Dataset/RGB"

splits = ["train", "val", "test"]

# CSV to store grams annotations
out_csv = os.path.join(ROOT, "fpb_grams_annotations.csv")
csv_rows = []

for split in splits:
    labels_dir = os.path.join(ROOT, split, "labels")
    pure_labels_dir = os.path.join(ROOT, split, "labels_yolo")
    os.makedirs(pure_labels_dir, exist_ok=True)

    for txt_path in glob.glob(os.path.join(labels_dir, "*.txt")):
        img_name = os.path.splitext(os.path.basename(txt_path))[0]

        with open(txt_path, "r") as f:
            lines = f.read().strip().splitlines()

        pure_lines = []
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 6:
                cls, xc, yc, w, h, grams = parts
            elif len(parts) == 5:
                # no grams – treat grams as None for that object
                cls, xc, yc, w, h = parts
                grams = None
            else:
                continue  # skip weird lines

            # 1) YOLO line: ONLY first 5 values
            pure_lines.append(" ".join([cls, xc, yc, w, h]))

            # 2) Store grams in CSV (if available)
            if grams is not None:
                csv_rows.append({
                    "split": split,
                    "image_id": img_name,
                    "object_id": line_idx,
                    "class_id": int(cls),
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": float(w),
                    "height": float(h),
                    "grams": float(grams)
                })

        # write pure YOLO label file
        out_txt = os.path.join(pure_labels_dir, os.path.basename(txt_path))
        with open(out_txt, "w") as f:
            f.write("\n".join(pure_lines))

# Write CSV with grams
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split", "image_id", "object_id",
            "class_id", "x_center", "y_center", "width", "height",
            "grams",
        ]
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print("Done. Pure YOLO labels in labels_yolo/, grams in fpb_grams_annotations.csv")
