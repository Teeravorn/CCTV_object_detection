# CCTV Vehicle Object Detection

ตรวจจับยานพาหนะ 8 ประเภท (Car, Motorcycle, Bus, Truck, Tuktuk, Van, Pickup, Songthaew) จากภาพกล้องวงจรปิดในกรุงเทพฯ

## Install

ใช้ Python 3.11 และ GPU ที่รองรับ CUDA

```bash
pip install -r requirements.txt
```

## Data

ดาวน์โหลดข้อมูลจาก Kaggle แล้ววางไว้ดังนี้

```
data_source/
├── train/        # ภาพ train
├── test/         # ภาพ test
└── train.csv     # annotation
sample_submission.csv
```

## Usage

```bash
# 1. ตรวจสอบข้อมูล แบ่ง train/valid ตามกล้อง, oversampling และแปลงเป็น COCO -> data/coco/
python prepare_data.py --oversample

# 2. เทรน (ผลลัพธ์อยู่ที่ runs/rfdetr_<variant>_...)
python train_rfdetr.py --variant 2xlarge --epochs 12 --aug strong --lr-schedule step --lr-drop 6 --patience 10

# 3. ทำนาย test set และสร้างไฟล์ส่ง Kaggle -> results/submission_rfdetr.csv
python predict.py --weights runs/<run_name>/checkpoint_best_total.pth
```

## Files

| Files | Description |
|---|---|
| `common.py` | path, การแบ่งกล้อง, ฟังก์ชันประเมินผลแบบเดียวกับ Kaggle |
| `analyze_data.py` | สรุปสถิติข้อมูลและปัญหาของ label |
| `find_duplicates.py` | หา label ที่ซ้อนทับกันแต่ต่างคลาส พร้อมสร้างภาพไว้ตรวจสอบ |
| `prepare_data.py` | ทำความสะอาด แบ่งข้อมูล oversampling และแปลงเป็น COCO |
| `augment.py` | ชุด augmentation แบบ strong |
| `train_rfdetr.py` | เทรน ประเมินผล และสร้างภาพตัวอย่าง |
| `detectors.py` | wrapper สำหรับการทำนาย และฟังก์ชัน dedupe |
| `predict.py` | ทำนาย test set และสร้างไฟล์ส่งผล |
| `evaluate.py` | เปรียบเทียบ checkpoint หลายตัวบน valid set |
| `train_pipeline_cmd.sh` | รวมคำสั่งที่ใช้ทดลอง |
