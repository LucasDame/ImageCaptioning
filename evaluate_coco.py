"""
Script d'évaluation COCO pour Image Captioning
===============================================

Version COCO de evaluate.py.

Différences avec Flickr8k :
- Le test set = val2017 officiel de COCO (pas de split manuel)
- Import depuis preprocessing_coco et config_coco
- Même logique d'évaluation BLEU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import numpy as np

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    print("NLTK non installé. Installation : pip install nltk")
    exit(1)

from utils.vocabulary import Vocabulary
from utils.preprocessing_coco import CaptionPreprocessor, ImagePreprocessor
from utils.data_loader import ImageCaptionDataset, CaptionCollate
from models.caption_model import load_model
from config_coco import CONFIG


class Evaluator:
    """
    Classe pour évaluer le modèle d'image captioning.
    Identique à la version Flickr8k.
    """

    def __init__(self, model, test_loader, vocabulary, device='cpu'):
        self.model      = model
        self.test_loader = test_loader
        self.vocabulary  = vocabulary
        self.device      = device

        self.model.to(device)
        self.model.eval()

        self.start_token = vocabulary.word2idx[vocabulary.start_token]
        self.end_token   = vocabulary.word2idx[vocabulary.end_token]
        self.pad_token   = vocabulary.word2idx[vocabulary.pad_token]

    def generate_captions(self, max_length=20):
        print("\nGénération des captions...")

        generated_captions  = []
        reference_captions  = []

        with torch.no_grad():
            for images, captions, lengths in tqdm(self.test_loader):
                images = images.to(self.device)

                batch_size = images.size(0)
                for i in range(batch_size):
                    image = images[i:i+1]

                    generated = self.model.generate_caption(
                        image,
                        max_length=max_length,
                        start_token=self.start_token,
                        end_token=self.end_token
                    )

                    generated_tokens = [
                        token.item() if torch.is_tensor(token) else token
                        for token in generated[0]
                        if (token.item() if torch.is_tensor(token) else token)
                        not in [self.start_token, self.end_token, self.pad_token]
                    ]
                    generated_text = self.vocabulary.denumericalize(generated_tokens)
                    generated_captions.append(generated_text.split())

                    reference_tokens = [
                        token.item() for token in captions[i]
                        if token.item() not in [self.start_token, self.end_token, self.pad_token]
                    ]
                    reference_text = self.vocabulary.denumericalize(reference_tokens)
                    reference_captions.append([reference_text.split()])

        return generated_captions, reference_captions

    def calculate_bleu_scores(self, generated_captions, reference_captions):
        print("\nCalcul des scores BLEU...")

        smooth = SmoothingFunction()

        bleu_scores = {
            'BLEU-1': corpus_bleu(reference_captions, generated_captions,
                                  weights=(1.0, 0, 0, 0),
                                  smoothing_function=smooth.method1),
            'BLEU-2': corpus_bleu(reference_captions, generated_captions,
                                  weights=(0.5, 0.5, 0, 0),
                                  smoothing_function=smooth.method1),
            'BLEU-3': corpus_bleu(reference_captions, generated_captions,
                                  weights=(0.33, 0.33, 0.33, 0),
                                  smoothing_function=smooth.method1),
            'BLEU-4': corpus_bleu(reference_captions, generated_captions,
                                  weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smooth.method1),
        }

        return bleu_scores

    def evaluate(self, max_length=20, num_examples=5):
        print("="*70)
        print("ÉVALUATION DU MODÈLE (COCO)")
        print("="*70)

        generated_captions, reference_captions = self.generate_captions(max_length)
        bleu_scores = self.calculate_bleu_scores(generated_captions, reference_captions)

        print("\n" + "="*70)
        print("SCORES BLEU")
        print("="*70)
        for metric, score in bleu_scores.items():
            print(f"{metric} : {score:.4f}")

        print("\n" + "="*70)
        print(f"EXEMPLES DE CAPTIONS GÉNÉRÉES (premiers {num_examples})")
        print("="*70)
        for i in range(min(num_examples, len(generated_captions))):
            print(f"\nExemple {i+1} :")
            print(f"  Référence : {' '.join(reference_captions[i][0])}")
            print(f"  Généré    : {' '.join(generated_captions[i])}")

        gen_lengths = [len(cap)    for cap in generated_captions]
        ref_lengths = [len(cap[0]) for cap in reference_captions]

        stats = {
            'bleu_scores':            bleu_scores,
            'num_samples':            len(generated_captions),
            'avg_generated_length':   np.mean(gen_lengths),
            'avg_reference_length':   np.mean(ref_lengths),
            'std_generated_length':   np.std(gen_lengths),
            'std_reference_length':   np.std(ref_lengths),
        }

        print("\n" + "="*70)
        print("STATISTIQUES")
        print("="*70)
        print(f"Nombre d'échantillons : {stats['num_samples']}")
        print(f"Longueur moyenne (générée)   : {stats['avg_generated_length']:.2f} ± {stats['std_generated_length']:.2f}")
        print(f"Longueur moyenne (référence) : {stats['avg_reference_length']:.2f} ± {stats['std_reference_length']:.2f}")

        return stats, generated_captions, reference_captions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de : {device}")

    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # ========================================================================
    # CHARGER LE MODÈLE
    # ========================================================================

    print("\nChargement du modèle...")
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"ERREUR : Le checkpoint {checkpoint_path} n'existe pas !")
        return

    model, info = load_model(
        checkpoint_path,
        device=device,
        encoder_type=CONFIG['encoder_type']
    )

    vocabulary = info['vocab']
    if vocabulary is None:
        if os.path.exists(CONFIG['vocab_path']):
            print(f"Chargement du vocabulaire depuis {CONFIG['vocab_path']}...")
            vocabulary = Vocabulary.load(CONFIG['vocab_path'])
        else:
            print(f"ERREUR : Vocabulaire introuvable !")
            return

    print(f"Taille du vocabulaire : {len(vocabulary)}")

    # ========================================================================
    # PRÉPARER LES DONNÉES DE TEST — val2017 officiel COCO
    # ========================================================================

    print("\nPréparation des données de test (COCO val2017)...")

    caption_prep = CaptionPreprocessor(
        CONFIG['val_captions_file'],
        CONFIG['val_images_dir']
    )
    test_pairs = caption_prep.get_image_caption_pairs()

    print(f"Nombre d'échantillons de test : {len(test_pairs)}")

    if len(test_pairs) == 0:
        print("ERREUR : Aucun échantillon de test trouvé !")
        return

    image_prep = ImagePreprocessor(image_size=CONFIG['image_size'], normalize=True)

    test_dataset = ImageCaptionDataset(
        test_pairs, vocabulary, image_prep, is_training=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=CaptionCollate(
            pad_idx=vocabulary.word2idx[vocabulary.pad_token]
        )
    )

    # ========================================================================
    # ÉVALUER
    # ========================================================================

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        vocabulary=vocabulary,
        device=device
    )

    stats, generated, references = evaluator.evaluate(
        max_length=CONFIG['max_caption_length'],
        num_examples=10
    )

    # ========================================================================
    # SAUVEGARDER LES RÉSULTATS
    # ========================================================================

    print("\nSauvegarde des résultats...")

    stats_path = os.path.join(CONFIG['results_dir'], 'evaluation_results_coco.json')
    with open(stats_path, 'w') as f:
        stats_json = {
            'bleu_scores':          stats['bleu_scores'],
            'num_samples':          int(stats['num_samples']),
            'avg_generated_length': float(stats['avg_generated_length']),
            'avg_reference_length': float(stats['avg_reference_length']),
            'std_generated_length': float(stats['std_generated_length']),
            'std_reference_length': float(stats['std_reference_length']),
        }
        json.dump(stats_json, f, indent=4)
    print(f"Résultats sauvegardés dans {stats_path}")

    examples = [
        {'id': i, 'reference': ' '.join(references[i][0]), 'generated': ' '.join(generated[i])}
        for i in range(min(50, len(generated)))
    ]
    examples_path = os.path.join(CONFIG['results_dir'], 'caption_examples_coco.json')
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=4)
    print(f"Exemples sauvegardés dans {examples_path}")

    print("\n" + "="*70)
    print("ÉVALUATION COCO TERMINÉE !")
    print("="*70)


if __name__ == "__main__":
    main()
