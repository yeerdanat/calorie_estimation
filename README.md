# calorie_estimation

YOLO + regression model for calorie estimation in a food tray.

The idea is simple: take a photo of a food tray, detect what food is on it, estimate how many grams of each food there is, and from that calculate the calories. Right now the detection part is working, the grams regression part is the next step.

## How it works

I use the Food Portion Benchmark (FPB) dataset, it is connected as a submodule. Its labels are not in the normal YOLO format — every line has 6 values, and the last one is the weight of the food in grams. YOLO expects only 5, so `fix_split.py` goes through all the labels, writes clean 5-value YOLO labels into separate `labels_yolo` folders, and collects all the grams into one csv file (`fpb_grams_annotations.csv`). That csv will be used later for training the regression that predicts grams.

Then `train.py` fine-tunes YOLOv8s on the dataset (50 epochs, image size 640, batch 16).

`main.py` is just a small check that cuda actually sees the gpu before training.

## Results

After 50 epochs on the RGB part of the dataset:

| metric | value |
|--------|-------|
| mAP50 | 0.83 |
| mAP50-95 | 0.74 |
| precision | 0.81 |
| recall | 0.77 |

Training curves and the confusion matrix are in `runs/detect/fpb_yolo9`.

## How to run

1. Install ultralytics: `pip install ultralytics`
2. Get the dataset: `git submodule update --init`
3. Fix the labels: `python fix_split.py`
4. Train: `python train.py`

The pretrained yolov8s weights download automatically on the first run.

## What's next

- regression head that predicts grams for each detected food
- mapping food class + grams to calories
- testing on real cafeteria trays
