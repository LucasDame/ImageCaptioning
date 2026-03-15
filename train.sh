python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

if [ ! -s "data/coco" ]; then
    chmod +x getCOCO.sh
    ./getCOCO.sh
fi

python3 prepare_data.py

python3 train.py --model densenet --scheduler plateau

python3 evaluate.py --checkpoint Checkpoint/densenet/cosine/best_model.pth

git add .

git commit -m "Update best model checkpoint densenet plateau and training logs"

git push

python3 train.py --model resnet --scheduler plateau

python3 evaluate.py --checkpoint Checkpoint/resnet/plateau/best_model.pth

git add .

git commit -m "Update best model checkpoint resnet plateau and training logs"

git push

python3 train.py --model densenet --scheduler cosine

python3 evaluate.py --checkpoint Checkpoint/densenet/cosine/best_model.pth

git add .

git commit -m "Update best model checkpoint densenet cosine and training logs"

git push

python3 train.py --model resnet --scheduler cosine

python3 evaluate.py --checkpoint Checkpoint/resnet/cosine/best_model.pth

git add .

git commit -m "Update best model checkpoint resnet cosine and training logs"

git push

python3 train.py --model cnn --scheduler plateau

python3 evaluate.py --checkpoint Checkpoint/cnn/plateau/best_model.pth

git add .

git commit -m "Update best model checkpoint cnn plateau and training logs"

git push