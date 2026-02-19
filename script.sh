source venv/bin/activate

pip install -r requirements.txt

python3 utils/data_downloader.py

python3 train.py

git switch -c update-best-model

git add checkpoints/best_model.pth

git commit -m "Update best model checkpoint"

git push origin/update-best-model