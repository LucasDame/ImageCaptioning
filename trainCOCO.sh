source venv/bin/activate

pip install -r requirements.txt

python3 train_coco.py

python3 train_coco2.py

git add checkpoints_coco/best_model.pth

git add checkpoints_coco2/best_model.pth

git commit -m "Update best model checkpoint"

git push