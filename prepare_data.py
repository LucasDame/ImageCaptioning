"""
prepare_data.py — Préparation du vocabulaire COCO
==================================================

À lancer UNE SEULE FOIS avant le premier entraînement.
Construit le vocabulaire depuis le train set COCO et le sauvegarde.

Utilisation :
    python prepare_data.py
    python prepare_data.py --freq_threshold 5   # seuil de fréquence (défaut: 5)
    python prepare_data.py --freq_threshold 10  # vocabulaire plus petit
"""

import argparse
import os

from config import BASE_CONFIG
from utils.vocabulary import Vocabulary
from utils.preprocessing import CaptionPreprocessor, ImagePreprocessor
from utils.data_loader import get_data_loaders
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description='Construit le vocabulaire COCO et teste les DataLoaders.'
    )
    parser.add_argument('--freq_threshold', type=int,
                        default=BASE_CONFIG['freq_threshold'],
                        help=f"Fréquence minimale d'un mot (défaut: {BASE_CONFIG['freq_threshold']})")
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Taille du batch pour le test (défaut: 4)')
    return parser.parse_args()


def main():
    args = parse_args()

    config = BASE_CONFIG

    print("="*70)
    print("PRÉPARATION DES DONNÉES COCO")
    print("="*70)

    # ── 1. Captions train ─────────────────────────────────────────────────────
    print("\n[1/4] Chargement des captions train...")
    train_prep = CaptionPreprocessor(
        captions_file=config['train_captions_file'],
        images_dir=config['train_images_dir']
    )
    all_captions = train_prep.get_all_captions()
    train_pairs  = train_prep.get_image_caption_pairs()
    print(f"  → {len(train_pairs)} paires image-caption (train)")
    print(f"  Exemple : '{all_captions[0]}'")

    # ── 2. Captions val ───────────────────────────────────────────────────────
    print("\n[2/4] Chargement des captions val...")
    val_prep  = CaptionPreprocessor(
        captions_file=config['val_captions_file'],
        images_dir=config['val_images_dir']
    )
    val_pairs = val_prep.get_image_caption_pairs()
    print(f"  → {len(val_pairs)} paires image-caption (val)")

    # ── 3. Vocabulaire ────────────────────────────────────────────────────────
    print(f"\n[3/4] Construction du vocabulaire (freq_threshold={args.freq_threshold})...")
    os.makedirs(os.path.dirname(config['vocab_path']), exist_ok=True)

    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(all_captions)
    vocab.save(config['vocab_path'])

    print(f"\n  Statistiques :")
    print(f"    Taille          : {len(vocab)} mots")
    print(f"    Tokens spéciaux : <PAD>(0), <START>(1), <END>(2), <UNK>(3)")
    print(f"    Seuil fréquence : {args.freq_threshold}")

    # Test de conversion
    test = "a dog is running in the park"
    indices = vocab.numericalize(test)
    print(f"\n  Test :")
    print(f"    Original      : '{test}'")
    print(f"    Indices       : {indices}")
    print(f"    Reconstruit   : '{vocab.denumericalize(indices)}'")

    # ── 4. Test DataLoader ────────────────────────────────────────────────────
    print(f"\n[4/4] Test des DataLoaders (batch_size={args.batch_size})...")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_prep = ImagePreprocessor(
        image_size=config['image_size'], normalize=False,
        train_transform=train_transform, val_transform=val_transform,
    )

    train_loader, val_loader = get_data_loaders(
        train_pairs=train_pairs, val_pairs=val_pairs,
        vocabulary=vocab, image_preprocessor=image_prep,
        batch_size=args.batch_size, num_workers=0,
        shuffle_train=True
    )

    images, captions, lengths = next(iter(train_loader))
    print(f"\n  Premier batch :")
    print(f"    images   : {images.shape}")
    print(f"    captions : {captions.shape}")
    print(f"    lengths  : {lengths.tolist()}")
    print(f"    Caption  : '{vocab.denumericalize(captions[0])}'")

    print("\n" + "="*70)
    print("PRÉPARATION TERMINÉE !")
    print("="*70)
    print(f"\nVocabulaire sauvegardé : {config['vocab_path']}")
    print("\nProchaines étapes :")
    print("  python train.py --model densenet --scheduler cosine")
    print("  python train.py --model resnet   --scheduler plateau")
    print("  python train.py --model cnn      --scheduler cosine")


if __name__ == "__main__":
    main()