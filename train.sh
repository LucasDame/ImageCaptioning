python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

if [ ! -s "data/coco" ]; then
    chmod +x getCOCO.sh
    ./getCOCO.sh
fi

python3 prepare_data.py

python train.py --model resnet --scheduler plateau --augdata False --label_smoothing 0.0

python3 evaluate.py --checkpoint Checkpoint/resnet/plateau/best_model.pth

git add .

git commit -m "Update best model checkpoint resnet plateau and training logs"

git push

python train.py --model densenet --scheduler plateau --augdata False --label_smoothing 0.0

python3 evaluate.py --checkpoint Checkpoint/densenet/plateau/best_model.pth

git add .

git commit -m "Update best model checkpoint densenet plateau and training logs"

git push

python train.py --model cnn --scheduler plateau --augdata False --label_smoothing 0.0

python3 evaluate.py --checkpoint Checkpoint/cnn/plateau/best_model.pth

git add .

git commit -m "Update best model checkpoint cnn plateau and training logs"

git push