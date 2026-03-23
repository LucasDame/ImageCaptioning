"""
DataLoader pour le dataset d'Image Captioning
Gère le chargement des batches d'images et captions avec padding
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class ImageCaptionDataset(Dataset):
    """
    Dataset PyTorch pour l'image captioning
    """
    
    def __init__(self, image_caption_pairs, vocabulary, image_preprocessor, is_training=True):
        """
        Args:
            image_caption_pairs (list): Liste de dicts {'image_path', 'caption', 'image_name'}
            vocabulary (Vocabulary): Instance du vocabulaire
            image_preprocessor (ImagePreprocessor): Préprocesseur d'images
            is_training (bool): Mode entraînement (avec augmentation)
        """
        self.pairs = image_caption_pairs
        self.vocab = vocabulary
        self.image_preprocessor = image_preprocessor
        self.is_training = is_training
    
    def __len__(self):
        """
        Retourne le nombre total de paires image-caption
        """
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Retourne une paire (image, caption) à l'index idx
        
        Returns:
            tuple: (image_tensor, caption_indices)
                - image_tensor: torch.Tensor de shape (3, H, W)
                - caption_indices: torch.Tensor de shape (seq_len,)
        """
        pair = self.pairs[idx]
        
        # Charger et prétraiter l'image
        image = self.image_preprocessor(pair['image_path'], self.is_training)
        
        # Convertir la caption en indices
        caption_indices = self.vocab.numericalize(pair['caption'])
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        
        return image, caption_tensor


class CaptionCollate:
    """
    Fonction de collate personnalisée pour gérer le padding des captions
    Les captions ont des longueurs différentes, il faut les padder au même length
    """
    
    def __init__(self, pad_idx):
        """
        Args:
            pad_idx (int): Index du token de padding (généralement 0)
        """
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        """
        Collate une liste de (image, caption) en un batch
        
        Args:
            batch (list): Liste de tuples (image_tensor, caption_tensor)
        
        Returns:
            tuple: (images, captions, lengths)
                - images: torch.Tensor de shape (batch_size, 3, H, W)
                - captions: torch.Tensor de shape (batch_size, max_seq_len)
                - lengths: torch.Tensor de shape (batch_size,) - longueurs réelles
        """
        # Séparer les images et les captions
        images = []
        captions = []
        
        for image, caption in batch:
            images.append(image)
            captions.append(caption)
        
        # Stack les images (toutes ont la même taille)
        images = torch.stack(images, dim=0)
        
        # Obtenir les longueurs avant padding
        lengths = torch.tensor([len(cap) for cap in captions])
        
        # Padder les captions à la longueur maximale du batch
        # pad_sequence attend une liste de tensors 1D et retourne un tensor 2D
        captions_padded = pad_sequence(
            captions, 
            batch_first=True,  # Shape: (batch_size, seq_len)
            padding_value=self.pad_idx
        )
        
        return images, captions_padded, lengths


def get_data_loaders(train_pairs, val_pairs, vocabulary, image_preprocessor, 
                     batch_size=32, num_workers=4, shuffle_train=True):
    """
    Crée les DataLoaders pour l'entraînement et la validation
    
    Args:
        train_pairs (list): Paires image-caption d'entraînement
        val_pairs (list): Paires image-caption de validation
        vocabulary (Vocabulary): Vocabulaire construit
        image_preprocessor (ImagePreprocessor): Préprocesseur d'images
        batch_size (int): Taille du batch
        num_workers (int): Nombre de workers pour le chargement
        shuffle_train (bool): Mélanger les données d'entraînement
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Créer les datasets
    train_dataset = ImageCaptionDataset(
        train_pairs, 
        vocabulary, 
        image_preprocessor, 
        is_training=True
    )
    
    val_dataset = ImageCaptionDataset(
        val_pairs, 
        vocabulary, 
        image_preprocessor, 
        is_training=False
    )
    
    # Index du token de padding
    pad_idx = vocabulary.word2idx[vocabulary.pad_token]
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=CaptionCollate(pad_idx=pad_idx),
        pin_memory=True  # Accélère le transfert CPU -> GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CaptionCollate(pad_idx=pad_idx),
        pin_memory=True
    )
    
    print(f"\nDataLoaders créés:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# Fonction pour tester le DataLoader
if __name__ == "__main__":
    from vocabulary import Vocabulary
    from preprocessing import ImagePreprocessor, CaptionPreprocessor
    
    print("=== Test DataLoader ===")
    
    # Exemple avec des données fictives
    # Dans la pratique, remplacer par vos vrais chemins
    
    # 1. Créer le vocabulaire
    captions = [
        "a dog is running",
        "two cats sitting",
        "the dog plays"
    ]
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(captions)
    
    # 2. Créer des paires fictives
    pairs = [
        {'image_path': 'data/Images/667626_18933d713e.jpg', 'caption': 'a dog is running', 'image_name': 'test.jpg'}
    ] * 10  # Répéter pour avoir plusieurs samples
    
    # 3. Créer le dataset
    image_prep = ImagePreprocessor()
    caption_prep = CaptionPreprocessor(captions_file="data/captions.txt", images_dir="data/Images")
    dataset = ImageCaptionDataset(pairs, vocab, image_prep, is_training=True)
    
    # 4. Créer le DataLoader
    pad_idx = vocab.word2idx[vocab.pad_token]
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=CaptionCollate(pad_idx=pad_idx)
    )
    
    # 5. Tester un batch
    print("\nTest d'un batch:")

    for images, captions, lengths in loader:
        print(f"Images shape: {images.shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"Lengths: {lengths}")
        break