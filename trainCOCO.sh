source venv/bin/activate

pip install -r requirements.txt

python3 train_coco.py

python3 train_coco2.py

git add .

git commit -m "Update best model checkpoint"

git push