"""
EXEMPLE PRATIQUE : Préparation des données COCO
================================================

Version COCO de prepare_data.py.

Différences avec Flickr8k :
- Les annotations sont dans deux fichiers JSON séparés (train / val)
- On utilise les splits OFFICIELS de COCO (train2017 / val2017)
  au lieu de faire un split manuel
- Le vocabulaire est construit uniquement sur le train set
"""

import torch
import os

from vocabulary import Vocabulary
from preprocessing_coco import ImagePreprocessor, CaptionPreprocessor
from data_loader import get_data_loaders


def main():
    """
    Pipeline complet de préparation des données COCO
    """

    # ========================================================================
    # 1. CONFIGURATION DES CHEMINS
    # ========================================================================

    TRAIN_CAPTIONS_FILE = "data/coco/annotations/captions_train2017.json"
    VAL_CAPTIONS_FILE   = "data/coco/annotations/captions_val2017.json"
    TRAIN_IMAGES_DIR    = "data/coco/train2017"
    VAL_IMAGES_DIR      = "data/coco/val2017"
    VOCAB_PATH          = "data/coco_vocab.pkl"

    print("="*70)
    print("PRÉPARATION DES DONNÉES COCO POUR IMAGE CAPTIONING")
    print("="*70)

    # ========================================================================
    # 2. CHARGER LES CAPTIONS D'ENTRAÎNEMENT
    # ========================================================================

    print("\n[ÉTAPE 1] Chargement des captions COCO train...")
    train_caption_processor = CaptionPreprocessor(
        captions_file=TRAIN_CAPTIONS_FILE,
        images_dir=TRAIN_IMAGES_DIR
    )

    all_captions = train_caption_processor.get_all_captions()
    train_pairs  = train_caption_processor.get_image_caption_pairs()
    print(f"  → {len(train_pairs)} paires image-caption (train)")
    print(f"\nExemple de caption : '{all_captions[0]}'")

    # ========================================================================
    # 3. CHARGER LES CAPTIONS DE VALIDATION
    # ========================================================================

    print("\n[ÉTAPE 2] Chargement des captions COCO val...")
    val_caption_processor = CaptionPreprocessor(
        captions_file=VAL_CAPTIONS_FILE,
        images_dir=VAL_IMAGES_DIR
    )
    val_pairs = val_caption_processor.get_image_caption_pairs()
    print(f"  → {len(val_pairs)} paires image-caption (val)")

    # ========================================================================
    # 4. CONSTRUIRE LE VOCABULAIRE (sur le train uniquement)
    # ========================================================================

    print("\n[ÉTAPE 3] Construction du vocabulaire...")
    vocabulary = Vocabulary(freq_threshold=5)
    vocabulary.build_vocabulary(all_captions)
    vocabulary.save(VOCAB_PATH)

    print(f"\n  Statistiques du vocabulaire :")
    print(f"    - Taille : {len(vocabulary)} mots")
    print(f"    - Tokens spéciaux : <PAD>(0), <START>(1), <END>(2), <UNK>(3)")

    # Test de conversion
    test_caption = "a dog is running in the park"
    indices = vocabulary.numericalize(test_caption)
    reconstructed = vocabulary.denumericalize(indices)
    print(f"\n  Test de conversion :")
    print(f"    Original      : '{test_caption}'")
    print(f"    Indices       : {indices}")
    print(f"    Reconstructed : '{reconstructed}'")

    # ========================================================================
    # 5. CRÉER LES DATALOADERS
    # ========================================================================

    print("\n[ÉTAPE 4] Création des DataLoaders...")

    image_preprocessor = ImagePreprocessor(image_size=224, normalize=True)

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

    images, captions, lengths = next(iter(train_loader))

    print(f"\n  Dimensions du batch :")
    print(f"    - Images   : {images.shape}")
    print(f"    - Captions : {captions.shape}")
    print(f"    - Lengths  : {lengths.shape}")

    first_caption_text = vocabulary.denumericalize(captions[0])
    print(f"\n  Première caption du batch :")
    print(f"    - Indices : {captions[0][:10].tolist()}...")
    print(f"    - Texte   : '{first_caption_text}'")
    print(f"    - Longueur réelle : {lengths[0].item()} mots")

    # ========================================================================
    # 7. INFORMATIONS SPÉCIFIQUES COCO
    # ========================================================================

    print("\n" + "="*70)
    print("INFORMATIONS COCO vs FLICKR8K")
    print("="*70)
    print("""
    FLICKR8K                        COCO
    --------                        ----
    8 000 images                    ~118 000 images train
    5 captions/image                5 captions/image
    40 000 paires total             ~591 000 paires train
    Format CSV                      Format JSON (annotations officiel)
    Split manuel (80/10/10)         Splits officiels train2017 / val2017
    Vocabulaire ~3 000 mots         Vocabulaire ~10 000 mots (freq>=5)

    STRUCTURE DES FICHIERS COCO :
    data/
      coco/
        annotations/
          captions_train2017.json   ← annotations train
          captions_val2017.json     ← annotations val
        train2017/                  ← ~118 000 images
        val2017/                    ← ~5 000 images
    """)

    print("\n" + "="*70)
    print("PRÉPARATION TERMINÉE AVEC SUCCÈS !")
    print("="*70)
    print(f"\nVocabulaire sauvegardé dans : {VOCAB_PATH}")
    print(f"Vous pouvez maintenant lancer : python train_coco.py")

    return vocabulary, train_loader, val_loader


if __name__ == "__main__":
    try:
        vocab, train_loader, val_loader = main()
    except FileNotFoundError as e:
        print(f"\nERREUR : {e}")
        print("\nAssurez-vous que :")
        print("  1. Le dataset COCO 2017 est téléchargé")
        print("  2. Les annotations sont dans data/coco/annotations/")
        print("  3. Les images sont dans data/coco/train2017/ et data/coco/val2017/")
        print("\nTéléchargement COCO :")
        print("  http://images.cocodataset.org/zips/train2017.zip")
        print("  http://images.cocodataset.org/zips/val2017.zip")
        print("  http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
