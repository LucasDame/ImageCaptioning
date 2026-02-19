"""
Téléchargeur de datasets depuis Kaggle
=======================================

Télécharge automatiquement les datasets populaires pour l'image captioning.
"""

import kagglehub
import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Télécharger un dataset depuis Kaggle")
parser.add_argument('--dataset', type=str, default="flickr8k", help="Nom du dataset à télécharger")

possibilities = ["flickr8k", "coco2014", "coco2017", "flicker30k", "all"]

def download_flickr8k():
     if not os.path.exists("./data/flicker8k/Images"):
        print("Téléchargement du dataset Flickr8k depuis Kaggle...")
        path = kagglehub.dataset_download("awsaf49/flickr8k-dataset", output_dir="./data/flicker8k")
        print("Dataset téléchargé dans:", path)

def download_coco2014():
    if not os.path.exists("./data/coco2014/images"):
        print("Téléchargement du dataset COCO 2014 depuis Kaggle...")
        path = kagglehub.dataset_download("awsaf49/coco-2014", output_dir="./data/coco2014")
        print("Dataset téléchargé dans:", path)

def download_coco2017():   
    if not os.path.exists("./data/coco2017/images"):
        print("Téléchargement du dataset COCO 2017 depuis Kaggle...")
        path = kagglehub.dataset_download("awsaf49/coco-2017-dataset", output_dir="./data/coco2017")
        print("Dataset téléchargé dans:", path)

def download_flickr30k():
    if not os.path.exists("./data/flickr30k/images"):
        print("Téléchargement du dataset Flickr30k depuis Kaggle...")
        path = kagglehub.dataset_download("awsaf49/flickr30k-dataset", output_dir="./data/flickr30k")
        print("Dataset téléchargé dans:", path)

def main():
    args = parser.parse_args()
    
    if args.dataset not in possibilities:
        print(f"Dataset inconnu: {args.dataset}")
        print(f"Choisissez parmi: {possibilities}")
        return
    
    if args.dataset == "flickr8k":
        download_flickr8k()
    elif args.dataset == "coco2014":
        download_coco2014()
    elif args.dataset == "coco2017":
        download_coco2017()
    elif args.dataset == "flickr30k":
        download_flickr30k()
    elif args.dataset == "all":
        download_flickr8k()
        download_coco2014()
        download_coco2017()
        download_flickr30k()

if __name__ == "__main__":
    main()
