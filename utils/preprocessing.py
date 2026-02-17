"""
Preprocessing pour les images et les captions
Gère les transformations d'images et la préparation des données
"""

import torch
from torchvision import transforms
from PIL import Image
import os
import json


class ImagePreprocessor:
    """
    Classe pour prétraiter les images
    """
    
    def __init__(self, image_size=224, normalize=True):
        """
        Args:
            image_size (int): Taille cible des images (carré)
            normalize (bool): Appliquer la normalisation ImageNet
        """
        self.image_size = image_size
        
        # Transformations pour l'entraînement
        transform_list = [
            transforms.Resize((image_size, image_size)),   
            transforms.ToTensor(),
        ]
        
        if normalize:
            # Normalisation ImageNet (standard pour les CNN)
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Moyenne RGB d'ImageNet
                    std=[0.229, 0.224, 0.225]     # Écart-type RGB d'ImageNet
                )
            )
        
        self.train_transform = transforms.Compose(transform_list)
        
        # Transformations pour la validation/test (sans augmentation)
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
        Charge et prétraite une image
        
        Args:
            image_path (str): Chemin vers l'image
            is_training (bool): Si True, applique l'augmentation
            
        Returns:
            torch.Tensor: Image prétraitée de shape (3, image_size, image_size)
        """
        image = Image.open(image_path).convert('RGB')
        
        if is_training:
            return self.train_transform(image)
        else:
            return self.val_transform(image)


class CaptionPreprocessor:
    """
    Classe pour charger et organiser les captions du dataset Flickr8k
    """
    
    def __init__(self, captions_file, images_dir):
        """
        Args:
            captions_file (str): Chemin vers le fichier de captions
                                Format: image_name.jpg\tcaption
            images_dir (str): Dossier contenant les images
        """
        self.captions_file = captions_file
        self.images_dir = images_dir
        self.image_caption_pairs = []
        self.all_captions = []
        
        self._load_captions()
    
    def _load_captions(self):
        """
        Charge les captions depuis le fichier texte de Flickr8k
        Format attendu: image_name.jpg\tcaption
        """
        print(f"Chargement des captions depuis {self.captions_file}")
        
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            # Séparer le nom de l'image et la caption
            parts = line.split(',')
            if len(parts) < 2:
                continue
            
            image_name = parts[0]
            caption = parts[1]
            
            # Chemin complet de l'image
            image_path = os.path.join(self.images_dir, image_name)
            
            # Vérifier que l'image existe
            if os.path.exists(image_path):
                self.image_caption_pairs.append({
                    'image_path': image_path,
                    'caption': caption,
                    'image_name': image_name
                })
                self.all_captions.append(caption)
        
        print(f"Chargé {len(self.image_caption_pairs)} paires image-caption")
        print(f"Nombre d'images uniques : {len(set([p['image_name'] for p in self.image_caption_pairs]))}")
    
    def get_all_captions(self):
        """
        Retourne toutes les captions (pour construire le vocabulaire)
        
        Returns:
            list: Liste de toutes les captions
        """
        return self.all_captions
    
    def get_image_caption_pairs(self):
        """
        Retourne les paires image-caption
        
        Returns:
            list: Liste de dictionnaires {'image_path', 'caption', 'image_name'}
        """
        return self.image_caption_pairs
    
    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        """
        Divise les données en train/val/test par image (pas par caption)
        
        Args:
            train_ratio (float): Ratio de données d'entraînement
            val_ratio (float): Ratio de données de validation
            
        Returns:
            dict: Dictionnaire contenant les listes 'train', 'val', 'test'
        """
        # Obtenir les images uniques
        unique_images = {}
        for pair in self.image_caption_pairs:
            img_name = pair['image_name']
            if img_name not in unique_images:
                unique_images[img_name] = []
            unique_images[img_name].append(pair)
        
        image_names = list(unique_images.keys())
        num_images = len(image_names)
        
        # Calculer les indices de split
        train_end = int(num_images * train_ratio)
        val_end = train_end + int(num_images * val_ratio)
        
        # Diviser les images
        train_images = image_names[:train_end]
        val_images = image_names[train_end:val_end]
        test_images = image_names[val_end:]
        
        # Créer les datasets
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for img_name in train_images:
            splits['train'].extend(unique_images[img_name])
        
        for img_name in val_images:
            splits['val'].extend(unique_images[img_name])
        
        for img_name in test_images:
            splits['test'].extend(unique_images[img_name])
        
        print(f"\nSplit des données:")
        print(f"  Train: {len(splits['train'])} paires ({len(train_images)} images)")
        print(f"  Val:   {len(splits['val'])} paires ({len(val_images)} images)")
        print(f"  Test:  {len(splits['test'])} paires ({len(test_images)} images)")
        
        return splits


# Fonction utilitaire pour tester
if __name__ == "__main__":
    # Test du préprocesseur d'images
    print("=== Test Image Preprocessor ===")
    # preprocessor = ImagePreprocessor(image_size=224)
    # image_tensor = preprocessor("data/Images/667626_18933d713e.jpg", is_training=True)
    # print(f"Image shape: {image_tensor.shape}")
    
    # Test du préprocesseur de captions
    print("\n=== Test Caption Preprocessor ===")
    caption_prep = CaptionPreprocessor(
        captions_file="data/captions.txt",
        images_dir="data/Images"
    )
    splits = caption_prep.split_data()