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
    Classe pour charger et prétraiter les images.

    RÔLE UNIQUE : ouvrir l'image depuis le disque et la convertir en PIL Image.
    Toute la logique de transform (resize, crop, augmentation, normalisation)
    est déléguée aux transforms externes passés par train_coco.py.

    Pourquoi ce refactoring ?
    ─────────────────────────
    L'ancienne version construisait ses propres transforms en interne (Resize +
    ToTensor + optionnellement Normalize), ce qui créait un conflit avec les
    transforms définis dans train_coco.py :

      Ancien flux (ERRONÉ) :
        PIL → [ImagePreprocessor] → Resize(224) + ToTensor → Tensor
            → [train_transform]  → Resize(256) sur Tensor  ← incohérent
                                 → ToTensor() sur Tensor   ← double conversion
                                 → Normalize()             ← appliquée sur mauvaises valeurs

      Nouveau flux (CORRECT) :
        path → [ImagePreprocessor.__call__] → PIL Image (brute, aucune transform)
             → [train_transform externe]   → Resize(256) + RandomCrop + ColorJitter
                                           + ToTensor + Normalize → Tensor final ✓

    Compatibilité ascendante :
    ──────────────────────────
    Si data_loader.py appelle image_preprocessor(path), il reçoit maintenant
    une PIL Image au lieu d'un Tensor. Si data_loader applique ensuite
    train_transform sur ce résultat, tout fonctionne correctement.

    Si data_loader attend un Tensor (ancienne interface) : passer
    normalize=True pour que le comportement par défaut reste compatible.
    """

    # Transforms par défaut (utilisés si aucun transform externe n'est fourni,
    # ou si data_loader n'accepte pas les paramètres train_transform/val_transform)
    _default_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    _default_val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, image_size=224, normalize=True,
                 train_transform=None, val_transform=None):
        """
        Args:
            image_size (int) : conservé pour compatibilité, non utilisé
                               si train_transform est fourni
            normalize (bool) : conservé pour compatibilité, non utilisé
                               si train_transform est fourni
            train_transform  : transform complet pour le train (PIL → Tensor)
                               Si None, utilise _default_train_transform
            val_transform    : transform complet pour la val   (PIL → Tensor)
                               Si None, utilise _default_val_transform
        """
        self.image_size = image_size

        # Priorité aux transforms externes (fournis par train_coco.py)
        # Fallback sur les defaults si non fournis
        self.train_transform = train_transform or self._default_train_transform
        self.val_transform   = val_transform   or self._default_val_transform

    def __call__(self, image_path, is_training=True):
        """
        Charge et prétraite une image.

        Args:
            image_path (str) : chemin vers le fichier image
            is_training (bool): True → train_transform, False → val_transform

        Returns:
            torch.Tensor : image prétraitée, shape (3, 224, 224)
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