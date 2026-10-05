# prepare data
python prepare_data.py 
python prepare_data.py --oversample

# training
python train_rfdetr.py --variant large --epochs 50 
python train_rfdetr.py --variant large --epochs 50 --aug strong


# evaluation
python evaluate_rfdetr.py --variant large --epochs 50

# prediction

python predict.py --weights runs/rfdetr_large/checkpoint_best_total.pth
python predict.py --weights runs/rfdetr_large/checkpoint_best_total.pth --dedupe-iou 0.9
