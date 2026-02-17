"""
Script d'évaluation pour Image Captioning
==========================================

Évalue le modèle sur le test set avec les métriques BLEU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np

# Importer NLTK pour les métriques BLEU
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    import nltk
    # Télécharger les données nécessaires
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    print("NLTK non installé. Installation: pip install nltk")
    exit(1)

# Nos modules
from utils import vocabulary, preprocessing, data_loader
from models import caption_model


class Evaluator:
    """
    Classe pour évaluer le modèle d'image captioning
    """
    
    def __init__(self, model, test_loader, vocabulary, device='cpu'):
        """
        Args:
            model (ImageCaptioningModel): Modèle entraîné
            test_loader (DataLoader): DataLoader de test
            vocabulary (Vocabulary): Vocabulaire
            device (str): 'cpu' ou 'cuda'
        """
        self.model = model
        self.test_loader = test_loader
        self.vocabulary = vocabulary
        self.device = device
        
        self.model.to(device)
        self.model.eval()
        
        # Tokens spéciaux
        self.start_token = vocabulary.word2idx[vocabulary.start_token]
        self.end_token = vocabulary.word2idx[vocabulary.end_token]
        self.pad_token = vocabulary.word2idx[vocabulary.pad_token]
    
    def generate_captions(self, max_length=20):
        """
        Génère des captions pour toutes les images du test set
        
        Args:
            max_length (int): Longueur maximale des captions
            
        Returns:
            tuple: (generated_captions, reference_captions)
        """
        print("\nGénération des captions...")
        
        generated_captions = []
        reference_captions = []
        
        with torch.no_grad():
            for images, captions, lengths in tqdm(self.test_loader):
                # Déplacer sur le device
                images = images.to(self.device)
                
                # Générer les captions
                batch_size = images.size(0)
                for i in range(batch_size):
                    # Image unique
                    image = images[i:i+1]
                    
                    # Générer la caption
                    generated = self.model.generate_caption(
                        image,
                        max_length=max_length,
                        start_token=self.start_token,
                        end_token=self.end_token
                    )
                    
                    # Convertir en texte (retirer les tokens spéciaux)
                    generated_tokens = [token for token in generated[0] 
                                       if token not in [self.start_token, self.end_token, self.pad_token]]
                    generated_text = self.vocabulary.denumericalize(generated_tokens)
                    generated_captions.append(generated_text.split())
                    
                    # Récupérer la référence (ground truth)
                    reference_tokens = [token.item() for token in captions[i] 
                                       if token.item() not in [self.start_token, self.end_token, self.pad_token]]
                    reference_text = self.vocabulary.denumericalize(reference_tokens)
                    reference_captions.append([reference_text.split()])  # Liste de listes pour BLEU
        
        return generated_captions, reference_captions
    
    def calculate_bleu_scores(self, generated_captions, reference_captions):
        """
        Calcule les scores BLEU
        
        Args:
            generated_captions (list): Captions générées (liste de listes de mots)
            reference_captions (list): Captions de référence (liste de listes de listes de mots)
        
        Returns:
            dict: Scores BLEU-1, BLEU-2, BLEU-3, BLEU-4
        """
        print("\nCalcul des scores BLEU...")
        
        # Fonction de smoothing pour éviter les scores de 0
        smooth = SmoothingFunction()
        
        # BLEU scores avec différents n-grams
        bleu_scores = {}
        
        # BLEU-1 (unigrams)
        bleu1 = corpus_bleu(
            reference_captions, 
            generated_captions,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smooth.method1
        )
        bleu_scores['BLEU-1'] = bleu1
        
        # BLEU-2 (bigrams)
        bleu2 = corpus_bleu(
            reference_captions,
            generated_captions,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth.method1
        )
        bleu_scores['BLEU-2'] = bleu2
        
        # BLEU-3 (trigrams)
        bleu3 = corpus_bleu(
            reference_captions,
            generated_captions,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smooth.method1
        )
        bleu_scores['BLEU-3'] = bleu3
        
        # BLEU-4 (4-grams)
        bleu4 = corpus_bleu(
            reference_captions,
            generated_captions,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth.method1
        )
        bleu_scores['BLEU-4'] = bleu4
        
        return bleu_scores
    
    def evaluate(self, max_length=20, num_examples=5):
        """
        Évaluation complète du modèle
        
        Args:
            max_length (int): Longueur maximale des captions
            num_examples (int): Nombre d'exemples à afficher
        
        Returns:
            dict: Résultats de l'évaluation
        """
        print("="*70)
        print("ÉVALUATION DU MODÈLE")
        print("="*70)
        
        # Générer les captions
        generated_captions, reference_captions = self.generate_captions(max_length)
        
        # Calculer les scores BLEU
        bleu_scores = self.calculate_bleu_scores(generated_captions, reference_captions)
        
        # Afficher les résultats
        print("\n" + "="*70)
        print("SCORES BLEU")
        print("="*70)
        for metric, score in bleu_scores.items():
            print(f"{metric}: {score:.4f}")
        
        # Afficher quelques exemples
        print("\n" + "="*70)
        print(f"EXEMPLES DE CAPTIONS GÉNÉRÉES (premiers {num_examples})")
        print("="*70)
        
        for i in range(min(num_examples, len(generated_captions))):
            print(f"\nExemple {i+1}:")
            print(f"  Référence: {' '.join(reference_captions[i][0])}")
            print(f"  Généré:    {' '.join(generated_captions[i])}")
        
        # Calculer des statistiques sur les longueurs
        gen_lengths = [len(cap) for cap in generated_captions]
        ref_lengths = [len(cap[0]) for cap in reference_captions]
        
        stats = {
            'bleu_scores': bleu_scores,
            'num_samples': len(generated_captions),
            'avg_generated_length': np.mean(gen_lengths),
            'avg_reference_length': np.mean(ref_lengths),
            'std_generated_length': np.std(gen_lengths),
            'std_reference_length': np.std(ref_lengths)
        }
        
        print("\n" + "="*70)
        print("STATISTIQUES")
        print("="*70)
        print(f"Nombre d'échantillons: {stats['num_samples']}")
        print(f"Longueur moyenne (générée):  {stats['avg_generated_length']:.2f} ± {stats['std_generated_length']:.2f}")
        print(f"Longueur moyenne (référence): {stats['avg_reference_length']:.2f} ± {stats['std_reference_length']:.2f}")
        
        return stats, generated_captions, reference_captions


def main():
    """
    Fonction principale pour l'évaluation
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    config = {
        'checkpoint_path': 'checkpoints/best_model.pth',
        'data_dir': 'data/flickr8k',
        'captions_file': 'data/flickr8k/captions.txt',
        'images_dir': 'data/flickr8k/Images',
        'results_dir': 'results',
        'encoder_type': 'lite',  # Doit correspondre au modèle entraîné
        'batch_size': 32,
        'num_workers': 4,
        'max_length': 20,
        'num_examples': 10
    }
    
    # Créer le dossier de résultats
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # ========================================================================
    # CHARGER LE MODÈLE
    # ========================================================================
    
    print("\nChargement du modèle...")
    
    if not os.path.exists(config['checkpoint_path']):
        print(f"ERREUR: Le checkpoint {config['checkpoint_path']} n'existe pas!")
        return
    
    model, info = load_model(
        config['checkpoint_path'],
        device=device,
        encoder_type=config['encoder_type']
    )
    
    # Charger le vocabulaire
    vocabulary = info['vocab']
    if vocabulary is None:
        vocab_path = 'data/vocab.pkl'
        if os.path.exists(vocab_path):
            print(f"Vocabulaire non trouvé dans le checkpoint, chargement depuis {vocab_path}...")
            vocabulary = Vocabulary.load(vocab_path)
        else:
            print(f"ERREUR: Vocabulaire introuvable dans le checkpoint et {vocab_path} n'existe pas!")
            return
    
    print(f"Taille du vocabulaire: {len(vocabulary)}")
    
    # ========================================================================
    # PRÉPARER LES DONNÉES DE TEST
    # ========================================================================
    
    print("\nPréparation des données de test...")
    
    if not os.path.exists(config['captions_file']):
        print(f"ERREUR: Le fichier {config['captions_file']} n'existe pas!")
        return
    
    if not os.path.exists(config['images_dir']):
        print(f"ERREUR: Le dossier {config['images_dir']} n'existe pas!")
        return
    
    caption_prep = CaptionPreprocessor(
        config['captions_file'],
        config['images_dir']
    )
    
    splits = caption_prep.split_data(train_ratio=0.8, val_ratio=0.1, seed=42)
    test_pairs = splits['test']
    
    print(f"Nombre d'échantillons de test: {len(test_pairs)}")
    
    if len(test_pairs) == 0:
        print("ERREUR: Aucun échantillon de test trouvé!")
        return
    
    # Créer le dataset et dataloader de test
    image_prep = ImagePreprocessor(image_size=224, normalize=True)
    
    test_dataset = ImageCaptionDataset(
        test_pairs,
        vocabulary,
        image_prep,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=CaptionCollate(pad_idx=vocabulary.word2idx[vocabulary.pad_token])
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
        max_length=config['max_length'],
        num_examples=config['num_examples']
    )
    
    # ========================================================================
    # SAUVEGARDER LES RÉSULTATS
    # ========================================================================
    
    print("\nSauvegarde des résultats...")
    
    # Sauvegarder les statistiques
    stats_path = os.path.join(config['results_dir'], 'evaluation_results.json')
    with open(stats_path, 'w') as f:
        # Convertir les numpy types en types Python natifs
        stats_json = {
            'bleu_scores': stats['bleu_scores'],
            'num_samples': int(stats['num_samples']),
            'avg_generated_length': float(stats['avg_generated_length']),
            'avg_reference_length': float(stats['avg_reference_length']),
            'std_generated_length': float(stats['std_generated_length']),
            'std_reference_length': float(stats['std_reference_length'])
        }
        json.dump(stats_json, f, indent=4)
    
    print(f"Résultats sauvegardés dans {stats_path}")
    
    # Sauvegarder quelques exemples
    examples = []
    for i in range(min(50, len(generated))):
        examples.append({
            'id': i,
            'reference': ' '.join(references[i][0]),
            'generated': ' '.join(generated[i])
        })
    
    examples_path = os.path.join(config['results_dir'], 'caption_examples.json')
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=4)
    
    print(f"Exemples sauvegardés dans {examples_path}")
    
    print("\n" + "="*70)
    print("ÉVALUATION TERMINÉE !")
    print("="*70)


if __name__ == "__main__":
    main()