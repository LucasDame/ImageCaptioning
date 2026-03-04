"""
Script d'évaluation COCO pour Image Captioning
===============================================

Compatible avec les trois encoder_type du nouveau modèle :
  'lite'      → EncoderCNNLite  + DecoderLSTM
  'full'      → EncoderCNN      + DecoderLSTM  (résiduel)
  'attention' → EncoderSpatial  + DecoderWithAttention

Le test set est le val2017 officiel de COCO.
La méthode de génération (greedy / beam_search) et le beam_width
sont lus depuis la config.
"""

import torch
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
from models2.caption_model2 import load_model
from config_coco import CONFIG


class Evaluator:

    def __init__(self, model, test_loader, vocabulary, config, device='cpu'):
        self.model       = model
        self.test_loader = test_loader
        self.vocabulary  = vocabulary
        self.config      = config
        self.device      = device

        self.model.to(device)
        self.model.eval()

        self.start_token = vocabulary.word2idx[vocabulary.start_token]
        self.end_token   = vocabulary.word2idx[vocabulary.end_token]
        self.pad_token   = vocabulary.word2idx[vocabulary.pad_token]

        self.method     = config.get('generation_method', 'greedy')
        self.beam_width = config.get('beam_width', 3)

    # -------------------------------------------------------------------------

    def generate_captions(self, max_length=20):
        print(f"\nGénération des captions (méthode : {self.method}"
              + (f", beam_width={self.beam_width})"
                 if self.method == 'beam_search' else ")"))

        generated_captions = []
        reference_captions = []

        with torch.no_grad():
            for images, captions, lengths in tqdm(self.test_loader):
                images = images.to(self.device)

                for i in range(images.size(0)):
                    image = images[i:i+1]

                    if self.method == 'beam_search':
                        generated = self.model.decoder.generate_beam_search(
                            self.model.encoder(image),
                            beam_width=self.beam_width,
                            max_length=max_length,
                            start_token=self.start_token,
                            end_token=self.end_token
                        )
                    else:
                        generated = self.model.generate_caption(
                            image,
                            max_length=max_length,
                            start_token=self.start_token,
                            end_token=self.end_token,
                            method='greedy'
                        )

                    # Convertir en texte
                    gen_tokens = [
                        t.item() if torch.is_tensor(t) else t
                        for t in generated[0]
                        if (t.item() if torch.is_tensor(t) else t)
                        not in [self.start_token, self.end_token, self.pad_token]
                    ]
                    generated_captions.append(
                        self.vocabulary.denumericalize(gen_tokens).split()
                    )

                    ref_tokens = [
                        t.item() for t in captions[i]
                        if t.item() not in [self.start_token,
                                            self.end_token, self.pad_token]
                    ]
                    reference_captions.append(
                        [self.vocabulary.denumericalize(ref_tokens).split()]
                    )

        return generated_captions, reference_captions

    # -------------------------------------------------------------------------

    def calculate_bleu_scores(self, generated, references):
        print("\nCalcul des scores BLEU...")
        smooth = SmoothingFunction()

        return {
            'BLEU-1': corpus_bleu(references, generated,
                                  weights=(1, 0, 0, 0),
                                  smoothing_function=smooth.method1),
            'BLEU-2': corpus_bleu(references, generated,
                                  weights=(.5, .5, 0, 0),
                                  smoothing_function=smooth.method1),
            'BLEU-3': corpus_bleu(references, generated,
                                  weights=(.33, .33, .33, 0),
                                  smoothing_function=smooth.method1),
            'BLEU-4': corpus_bleu(references, generated,
                                  weights=(.25, .25, .25, .25),
                                  smoothing_function=smooth.method1),
        }

    # -------------------------------------------------------------------------

    def evaluate(self, max_length=20, num_examples=5):
        print("="*70)
        print("ÉVALUATION DU MODÈLE (COCO)")
        print("="*70)

        generated, references = self.generate_captions(max_length)
        bleu_scores = self.calculate_bleu_scores(generated, references)

        print("\n" + "="*70)
        print("SCORES BLEU")
        print("="*70)
        for metric, score in bleu_scores.items():
            print(f"  {metric} : {score:.4f}")

        print("\n" + "="*70)
        print(f"EXEMPLES (premiers {num_examples})")
        print("="*70)
        for i in range(min(num_examples, len(generated))):
            print(f"\nExemple {i+1} :")
            print(f"  Référence : {' '.join(references[i][0])}")
            print(f"  Généré    : {' '.join(generated[i])}")

        gen_lengths = [len(c)    for c in generated]
        ref_lengths = [len(c[0]) for c in references]

        stats = {
            'bleu_scores':          bleu_scores,
            'num_samples':          len(generated),
            'avg_generated_length': np.mean(gen_lengths),
            'avg_reference_length': np.mean(ref_lengths),
            'std_generated_length': np.std(gen_lengths),
            'std_reference_length': np.std(ref_lengths),
        }

        print("\n" + "="*70)
        print("STATISTIQUES")
        print("="*70)
        print(f"  Échantillons : {stats['num_samples']}")
        print(f"  Longueur moy. générée   : "
              f"{stats['avg_generated_length']:.2f} ± "
              f"{stats['std_generated_length']:.2f}")
        print(f"  Longueur moy. référence : "
              f"{stats['avg_reference_length']:.2f} ± "
              f"{stats['std_reference_length']:.2f}")

        return stats, generated, references


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de : {device}")

    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # -------------------------------------------------------------------------
    # CHARGER LE MODÈLE
    # -------------------------------------------------------------------------

    print("\nChargement du modèle...")
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"ERREUR : checkpoint introuvable → {checkpoint_path}")
        return

    # load_model détecte automatiquement l'encoder_type depuis le checkpoint
    model, info = load_model(checkpoint_path, device=device)

    vocabulary = info['vocab']
    if vocabulary is None:
        if os.path.exists(CONFIG['vocab_path']):
            print(f"Chargement du vocabulaire depuis {CONFIG['vocab_path']}...")
            vocabulary = Vocabulary.load(CONFIG['vocab_path'])
        else:
            print("ERREUR : vocabulaire introuvable !")
            return

    print(f"Taille du vocabulaire : {len(vocabulary)}")

    # -------------------------------------------------------------------------
    # DONNÉES DE TEST — val2017 officiel
    # -------------------------------------------------------------------------

    print("\nPréparation des données de test (COCO val2017)...")
    caption_prep = CaptionPreprocessor(
        CONFIG['val_captions_file'],
        CONFIG['val_images_dir']
    )
    test_pairs = caption_prep.get_image_caption_pairs()
    print(f"Échantillons de test : {len(test_pairs)}")

    if not test_pairs:
        print("ERREUR : aucun échantillon de test !")
        return

    image_prep   = ImagePreprocessor(image_size=CONFIG['image_size'], normalize=True)
    test_dataset = ImageCaptionDataset(
        test_pairs, vocabulary, image_prep, is_training=False
    )
    test_loader  = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=CaptionCollate(
            pad_idx=vocabulary.word2idx[vocabulary.pad_token]
        )
    )

    # -------------------------------------------------------------------------
    # ÉVALUATION
    # -------------------------------------------------------------------------

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        vocabulary=vocabulary,
        config=CONFIG,
        device=device
    )

    stats, generated, references = evaluator.evaluate(
        max_length=CONFIG['max_caption_length'],
        num_examples=10
    )

    # -------------------------------------------------------------------------
    # SAUVEGARDE
    # -------------------------------------------------------------------------

    print("\nSauvegarde des résultats...")

    stats_path = os.path.join(CONFIG['results_dir'],
                              'evaluation_results_coco.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'encoder_type':         CONFIG['encoder_type'],
            'generation_method':    CONFIG['generation_method'],
            'bleu_scores':          stats['bleu_scores'],
            'num_samples':          int(stats['num_samples']),
            'avg_generated_length': float(stats['avg_generated_length']),
            'avg_reference_length': float(stats['avg_reference_length']),
            'std_generated_length': float(stats['std_generated_length']),
            'std_reference_length': float(stats['std_reference_length']),
        }, f, indent=4)
    print(f"Résultats → {stats_path}")

    examples_path = os.path.join(CONFIG['results_dir'],
                                 'caption_examples_coco.json')
    with open(examples_path, 'w') as f:
        json.dump([
            {'id': i,
             'reference': ' '.join(references[i][0]),
             'generated': ' '.join(generated[i])}
            for i in range(min(50, len(generated)))
        ], f, indent=4)
    print(f"Exemples   → {examples_path}")

    print("\n" + "="*70)
    print("ÉVALUATION COCO TERMINÉE !")
    print("="*70)


if __name__ == "__main__":
    main()
