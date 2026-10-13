# prepare data
python prepare_data.py 
python prepare_data.py --oversample

# training
python train_rfdetr.py --variant large --epochs 50 --batch auto
python train_rfdetr.py --variant large --epochs 50 --aug strong
python train_rfdetr.py --variant large --epochs 50 --aug strong --lr-schedule step --lr-drop 6 --warmup-epochs 0 --patience 15 --batch 8 --seed 42
python train_rfdetr.py --variant large --epochs 12 --aug strong --lr-schedule cosine --warmup-epochs 1 --min-lr-factor 0.05 --batch 8 --seed 42

# evaluation
python evaluate.py --rfdetr runs/rfdetr_large/checkpoint_best_ema.pth runs/rfdetr_large/checkpoint_best_total.pth

# prediction

python predict.py --weights runs/rfdetr_large_os/checkpoint_best_total.pth
python predict.py --weights runs/rfdetr_large_os/checkpoint_best_total.pth --dedupe-iou 0.9
