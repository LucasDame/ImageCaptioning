"""
Preprocessing COCO pour les images et les captions
====================================================

Version COCO de preprocessing.py.
Différence principale : les captions sont dans un fichier JSON (format MS-COCO)
au lieu d'un CSV tabulé comme dans Flickr8k.
"""

import json
import random
import torch
from torchvision import transforms
from PIL import Image
import os


class ImagePreprocessor:
    """
    Classe pour prétraiter les images.
    Identique à la version Flickr8k.
    """

    def __init__(self, image_size=224, normalize=True):
        """
        Args:
            image_size (int): Taille cible des images (carré)
            normalize (bool): Appliquer la normalisation ImageNet
        """
        self.image_size = image_size

        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        self.train_transform = transforms.Compose(transform_list)

        val_transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            val_transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        self.val_transform = transforms.Compose(val_transform_list)

    def __call__(self, image_path, is_training=True):
        """
        Charge et prétraite une image.

        Args:
            image_path (str): Chemin vers l'image
            is_training (bool): Si True, applique l'augmentation

        Returns:
            torch.Tensor: Image prétraitée de shape (3, H, W)
        """
        image = Image.open(image_path).convert('RGB')

        if is_training:
            return self.train_transform(image)
        else:
            return self.val_transform(image)


class CaptionPreprocessor:
    """
    Classe pour charger et organiser les captions du dataset COCO.

    Différence avec Flickr8k :
    - Flickr8k : fichier CSV  ->  image_name.jpg,caption
    - COCO     : fichier JSON ->  format MS-COCO officiel
                 (clés 'images', 'annotations')
    """

    def __init__(self, captions_file, images_dir):
        """
        Args:
            captions_file (str): Chemin vers le fichier JSON d'annotations COCO
                                 Ex: 'data/coco/annotations/captions_train2017.json'
            images_dir (str): Dossier contenant les images
                              Ex: 'data/coco/train2017'
        """
        self.captions_file = captions_file
        self.images_dir = images_dir
        self.image_caption_pairs = []
        self.all_captions = []

        self._load_captions()

    def _load_captions(self):
        """
        Charge les captions depuis le fichier JSON COCO.

        Structure du JSON :
        {
            "images": [
                {"id": 391895, "file_name": "000000391895.jpg", ...},
                ...
            ],
            "annotations": [
                {"image_id": 391895, "caption": "a man ...", "id": 1},
                ...
            ]
        }
        """
        print(f"Chargement des captions COCO depuis {self.captions_file}")

        with open(self.captions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Construire un dictionnaire image_id -> file_name
        id_to_filename = {
            img['id']: img['file_name']
            for img in data['images']
        }

        skipped = 0
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption  = ann['caption'].strip()
            image_name = id_to_filename.get(image_id)

            if image_name is None:
                skipped += 1
                continue

            image_path = os.path.join(self.images_dir, image_name)

            if os.path.exists(image_path):
                self.image_caption_pairs.append({
                    'image_path':  image_path,
                    'caption':     caption,
                    'image_name':  image_name
                })
                self.all_captions.append(caption)
            else:
                skipped += 1

        print(f"Chargé {len(self.image_caption_pairs)} paires image-caption")
        print(f"Images uniques : {len(set(p['image_name'] for p in self.image_caption_pairs))}")
        if skipped:
            print(f"Ignorées (image introuvable) : {skipped}")

    def get_all_captions(self):
        """
        Retourne toutes les captions (pour construire le vocabulaire).

        Returns:
            list: Liste de toutes les captions
        """
        return self.all_captions

    def get_image_caption_pairs(self):
        """
        Retourne les paires image-caption.

        Returns:
            list: Liste de dicts {'image_path', 'caption', 'image_name'}
        """
        return self.image_caption_pairs

    def split_data(self, train_ratio=0.8, val_ratio=0.1, random_seed=42):
        """
        Divise les données en train/val/test PAR IMAGE (pas par caption).

        Note : pour COCO, il est recommandé d'utiliser les splits officiels
        (train2017 / val2017) via deux instances séparées de CaptionPreprocessor
        plutôt que cette méthode. Elle est conservée ici par compatibilité.

        Args:
            train_ratio (float): Ratio d'entraînement
            val_ratio (float): Ratio de validation
            random_seed (int): Seed aléatoire

        Returns:
            dict: {'train': [...], 'val': [...], 'test': [...]}
        """
        # Regrouper les paires par image
        unique_images = {}
        for pair in self.image_caption_pairs:
            img_name = pair['image_name']
            if img_name not in unique_images:
                unique_images[img_name] = []
            unique_images[img_name].append(pair)

        image_names = sorted(unique_images.keys())
        num_images  = len(image_names)

        random.seed(random_seed)
        random.shuffle(image_names)

        train_end = int(num_images * train_ratio)
        val_end   = train_end + int(num_images * val_ratio)

        train_images = image_names[:train_end]
        val_images   = image_names[train_end:val_end]
        test_images  = image_names[val_end:]

        splits = {'train': [], 'val': [], 'test': []}

        for img_name in train_images:
            splits['train'].extend(unique_images[img_name])
        for img_name in val_images:
            splits['val'].extend(unique_images[img_name])
        for img_name in test_images:
            splits['test'].extend(unique_images[img_name])

        print(f"\nSplit des données :")
        print(f"  Train : {len(splits['train'])} paires ({len(train_images)} images)")
        print(f"  Val   : {len(splits['val'])} paires ({len(val_images)} images)")
        print(f"  Test  : {len(splits['test'])} paires ({len(test_images)} images)")

        return splits


# Fonction utilitaire pour tester
if __name__ == "__main__":
    print("=== Test CaptionPreprocessor COCO ===")
    caption_prep = CaptionPreprocessor(
        captions_file="data/coco/annotations/captions_train2017.json",
        images_dir="data/coco/train2017"
    )
    captions = caption_prep.get_all_captions()
    print(f"Exemple de caption : '{captions[0]}'")
