"""
EXEMPLE PRATIQUE: Utilisation complète du preprocessing et dataloader
======================================================================

Ce script montre comment utiliser tous les composants ensemble:
1. Charger les données Flickr8k
2. Construire le vocabulaire
3. Créer les DataLoaders
4. Tester un batch
"""

import torch
import sys
import os

# Importer nos modules
from vocabulary import Vocabulary
from preprocessing import ImagePreprocessor, CaptionPreprocessor
from data_loader import get_data_loaders


def main():
    """
    Pipeline complet de préparation des données
    """
    
    # ========================================================================
    # 1. CONFIGURATION DES CHEMINS
    # ========================================================================
    
    # IMPORTANT: Adapter ces chemins à votre structure de projet
    DATA_DIR = "data"
    CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
    IMAGES_DIR = os.path.join(DATA_DIR, "Images")
    VOCAB_PATH = "data/vocab.pkl"
    
    print("="*70)
    print("PRÉPARATION DES DONNÉES POUR IMAGE CAPTIONING")
    print("="*70)
    
    
    # ========================================================================
    # 2. CHARGER ET PRÉTRAITER LES CAPTIONS
    # ========================================================================
    
    print("\n[ÉTAPE 1] Chargement des captions...")
    caption_processor = CaptionPreprocessor(
        captions_file=CAPTIONS_FILE,
        images_dir=IMAGES_DIR
    )
    
    # Obtenir toutes les captions pour construire le vocabulaire
    all_captions = caption_processor.get_all_captions()
    print(f"  → {len(all_captions)} captions chargées")
    
    # Exemple de caption
    print(f"\nExemple de caption: '{all_captions[0]}'")
    
    
    # ========================================================================
    # 3. CONSTRUIRE LE VOCABULAIRE
    # ========================================================================
    
    print("\n[ÉTAPE 2] Construction du vocabulaire...")
    vocabulary = Vocabulary(freq_threshold=5)
    vocabulary.build_vocabulary(all_captions)
    
    # Sauvegarder le vocabulaire
    vocabulary.save(VOCAB_PATH)
    
    # Afficher quelques statistiques
    print(f"\n  Statistiques du vocabulaire:")
    print(f"    - Taille: {len(vocabulary)} mots")
    print(f"    - Tokens spéciaux: <PAD>(0), <START>(1), <END>(2), <UNK>(3)")
    
    # Test de conversion
    test_caption = "a dog is running in the park"
    indices = vocabulary.numericalize(test_caption)
    reconstructed = vocabulary.denumericalize(indices)
    
    print(f"\n  Test de conversion:")
    print(f"    Original:      '{test_caption}'")
    print(f"    Indices:       {indices}")
    print(f"    Reconstructed: '{reconstructed}'")
    
    
    # ========================================================================
    # 4. DIVISER LES DONNÉES (TRAIN/VAL/TEST)
    # ========================================================================
    
    print("\n[ÉTAPE 3] Division des données...")
    splits = caption_processor.split_data(
        train_ratio=0.8,   # 80% pour l'entraînement
        val_ratio=0.1      # 10% pour la validation, 10% pour le test
    )
    
    train_pairs = splits['train']
    val_pairs = splits['val']
    test_pairs = splits['test']
    
    
    # ========================================================================
    # 5. CRÉER LES DATALOADERS
    # ========================================================================
    
    print("\n[ÉTAPE 4] Création des DataLoaders...")
    
    # Préprocesseur d'images
    image_preprocessor = ImagePreprocessor(
        image_size=224,
        normalize=True
    )
    
    # Créer les DataLoaders
    train_loader, val_loader = get_data_loaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        vocabulary=vocabulary,
        image_preprocessor=image_preprocessor,
        batch_size=32,
        num_workers=4,
        shuffle_train=True
    )
    
    
    # ========================================================================
    # 6. TESTER UN BATCH
    # ========================================================================
    
    print("\n[ÉTAPE 5] Test d'un batch...")
    
    # Récupérer un batch du train_loader
    images, captions, lengths = next(iter(train_loader))
    
    print(f"\n  Dimensions du batch:")
    print(f"    - Images:   {images.shape}")
    print(f"    - Captions: {captions.shape}")
    print(f"    - Lengths:  {lengths.shape}")
    
    print(f"\n  Détails:")
    print(f"    - Batch size:        {images.shape[0]}")
    print(f"    - Image channels:    {images.shape[1]}")
    print(f"    - Image size:        {images.shape[2]}x{images.shape[3]}")
    print(f"    - Max caption length: {captions.shape[1]}")
    
    # Afficher la première caption du batch
    first_caption_indices = captions[0]
    first_caption_text = vocabulary.denumericalize(first_caption_indices)
    
    print(f"\n  Première caption du batch:")
    print(f"    - Indices: {first_caption_indices[:10].tolist()}...")
    print(f"    - Texte:   '{first_caption_text}'")
    print(f"    - Longueur réelle: {lengths[0].item()} mots")
    
    
    # ========================================================================
    # 7. COMPRENDRE LE FORMAT DES DONNÉES
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPLICATIONS - FORMAT DES DONNÉES")
    print("="*70)
    
    print("""
    IMAGES:
    -------
    Shape: (batch_size, channels, height, width)
    - batch_size: Nombre d'images dans le batch (32)
    - channels: 3 (RGB)
    - height, width: 224x224 (après preprocessing)
    - Valeurs: Normalisées avec mean/std d'ImageNet
    
    CAPTIONS:
    ---------
    Shape: (batch_size, max_seq_len)
    - batch_size: Nombre de captions (32)
    - max_seq_len: Longueur de la caption la plus longue du batch
    - Valeurs: Indices de mots (0 à vocab_size-1)
    - Padding: Les captions courtes sont paddées avec 0 (<PAD>)
    
    LENGTHS:
    --------
    Shape: (batch_size,)
    - Contient la longueur réelle de chaque caption (avant padding)
    - Utile pour ignorer le padding dans les calculs de loss
    
    EXEMPLE DE CAPTION PADDÉE:
    --------------------------
    Caption originale: "a dog is running"
    Indices: [1, 45, 123, 67, 89, 2]  (longueur: 6)
    
    Dans un batch où max_len=10:
    Indices paddés: [1, 45, 123, 67, 89, 2, 0, 0, 0, 0]
                     ^                    ^
                     <START>               <END> puis <PAD>
    """)
    
    
    # ========================================================================
    # 8. UTILISATION POUR L'ENTRAÎNEMENT
    # ========================================================================
    
    print("\n" + "="*70)
    print("UTILISATION DANS LA BOUCLE D'ENTRAÎNEMENT")
    print("="*70)
    
    print("""
    for epoch in range(num_epochs):
        for batch_idx, (images, captions, lengths) in enumerate(train_loader):
            
            # Transférer sur GPU si disponible
            images = images.to(device)
            captions = captions.to(device)
            
            # 1. Forward pass de l'encoder
            features = encoder(images)  # (batch_size, feature_dim)
            
            # 2. Préparer input/target pour teacher forcing
            inputs = captions[:, :-1]   # Tous sauf <END>
            targets = captions[:, 1:]   # Tous sauf <START>
            
            # 3. Forward pass du decoder
            # Le decoder convertit automatiquement les indices en embeddings
            outputs = decoder(features, inputs)
            
            # 4. Calculer la loss
            loss = criterion(outputs.reshape(-1, vocab_size), 
                           targets.reshape(-1))
            
            # 5. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """)
    
    print("\n" + "="*70)
    print("PRÉPARATION TERMINÉE AVEC SUCCÈS !")
    print("="*70)
    print(f"\nVocabulaire sauvegardé dans: {VOCAB_PATH}")
    print(f"Vous pouvez maintenant:")
    print(f"  1. Implémenter l'Encoder (CNN)")
    print(f"  2. Implémenter le Decoder (LSTM avec embeddings)")
    print(f"  3. Lancer l'entraînement")
    
    return vocabulary, train_loader, val_loader


# ========================================================================
# FONCTION POUR VISUALISER DES EXEMPLES
# ========================================================================

def visualize_batch_samples(train_loader, vocabulary, num_samples=3):
    """
    Affiche quelques exemples du dataset
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Récupérer un batch
    images, captions, lengths = next(iter(train_loader))
    
    # Dénormaliser les images pour l'affichage
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i in range(min(num_samples, images.shape[0])):
        # Dénormaliser
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Récupérer la caption
        caption_text = vocabulary.denumericalize(captions[i])
        
        # Afficher
        axes[i].imshow(img)
        axes[i].set_title(f"Caption: {caption_text}", fontsize=10, wrap=True)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/batch_samples.png')
    print("\nÉchantillons sauvegardés dans data/batch_samples.png")


if __name__ == "__main__":
    # Exécuter le pipeline complet
    try:
        vocab, train_loader, val_loader = main()
        
        # Optionnel: visualiser des exemples
        # visualize_batch_samples(train_loader, vocab)
        
    except FileNotFoundError as e:
        print(f"\nERREUR: {e}")
        print("\nAssurez-vous que:")
        print("  1. Le dataset Flickr8k est téléchargé")
        print("  2. Les chemins dans le script sont corrects")
        print("  3. Le fichier captions.txt existe")
        print("  4. Le dossier Images/ contient les images")