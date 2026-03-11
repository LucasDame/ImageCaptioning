source venv/bin/activate

pip install -r requirements.txt

python3 train_coco3.py

git add .

git commit -m "Update best model checkpoint densenet"

git push

python3 train_coco2.py

git add .

git commit -m "Update best model checkpoint resnet"

git push