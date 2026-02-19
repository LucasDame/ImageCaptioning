"""
Script de Démo pour Image Captioning
=====================================

Génère une caption pour une image fournie.
Parfait pour la présentation finale !
"""

import torch
import argparse
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Force l'utilisation d'une fenêtre interactive
import matplotlib.pyplot as plt
import os

# Nos modules
from utils.vocabulary import Vocabulary
from utils.preprocessing import ImagePreprocessor
from models.caption_model import load_model
from config import CONFIG


class CaptionDemo:
    """
    Classe pour faire des démos de génération de captions
    """
    
    def __init__(self, model_path, vocab_path=None, encoder_type='lite'):
        """
        Args:
            model_path (str): Chemin vers le modèle entraîné
            vocab_path (str): Chemin vers le vocabulaire (optionnel si dans le checkpoint)
            encoder_type (str): Type d'encoder ('full' ou 'lite')
        """
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de: {self.device}")
        
        # Charger le modèle
        print(f"\nChargement du modèle depuis {model_path}...")
        self.model, info = load_model(model_path, device=self.device, encoder_type=encoder_type)
        
        # Charger le vocabulaire
        self.vocabulary = info['vocab']
        if self.vocabulary is None and vocab_path is not None:
            print(f"Chargement du vocabulaire depuis {vocab_path}...")
            self.vocabulary = Vocabulary.load(vocab_path)
        elif self.vocabulary is None:
            raise ValueError("Vocabulaire non trouvé. Spécifiez vocab_path.")
        
        # Préprocesseur d'images
        self.image_preprocessor = ImagePreprocessor(image_size=CONFIG['image_size'], normalize=True)
        
        # Tokens
        self.start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        self.end_token = self.vocabulary.word2idx[self.vocabulary.end_token]
        self.pad_token = self.vocabulary.word2idx[self.vocabulary.pad_token]
        
        print("✓ Modèle chargé et prêt !")
    
    def generate_caption(self, image_path, max_length=20, method='greedy'):
        """
        Génère une caption pour une image
        
        Args:
            image_path (str): Chemin vers l'image
            max_length (int): Longueur maximale de la caption
            method (str): 'greedy' ou 'beam_search'
        
        Returns:
            str: Caption générée
        """
        # Vérifier que l'image existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image non trouvée: {image_path}")
        
        # Charger et préprocesser l'image
        image = self.image_preprocessor(image_path, is_training=False)
        image = image.unsqueeze(0).to(self.device)  # Ajouter batch dimension
        
        # Générer la caption
        self.model.eval()
        with torch.no_grad():
            caption_indices = self.model.generate_caption(
                image,
                max_length=max_length,
                start_token=self.start_token,
                end_token=self.end_token,
                method=method
            )
        
        # Convertir les tensors en entiers Python
        caption_tokens = []
        for token in caption_indices[0]:
            token_val = token.item() if torch.is_tensor(token) else token
            # Filtrer les tokens spéciaux
            if token_val not in [self.start_token, self.end_token, self.pad_token]:
                caption_tokens.append(token_val)
        
        # Convertir en texte
        caption = self.vocabulary.denumericalize(caption_tokens)
        
        return caption
    
    def display_result(self, image_path, caption, save_path=None):
        """
        Affiche l'image avec sa caption
        
        Args:
            image_path (str): Chemin vers l'image
            caption (str): Caption générée
            save_path (str): Chemin pour sauvegarder (optionnel)
        """
        # Charger l'image
        img = Image.open(image_path).convert('RGB')
        
        # Créer la figure
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Ajouter la caption en titre
        plt.title(f'Caption: "{caption}"', 
                 fontsize=16, 
                 fontweight='bold',
                 pad=20,
                 wrap=True)
        
        plt.tight_layout()
        
        # Sauvegarder si demandé
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Résultat sauvegardé dans {save_path}")
        
        plt.show(block = True)
    
    def demo_single_image(self, image_path, max_length=20, save_path=None):
        """
        Démo complète pour une seule image
        
        Args:
            image_path (str): Chemin vers l'image
            max_length (int): Longueur maximale de la caption
            save_path (str): Chemin pour sauvegarder (optionnel)
        """
        print("\n" + "="*70)
        print("GÉNÉRATION DE CAPTION")
        print("="*70)
        print(f"\nImage: {image_path}")
        
        # Générer la caption
        print("\nGénération en cours...")
        caption = self.generate_caption(image_path, max_length=max_length)
        
        # Afficher le résultat
        print(f"\nCaption générée: \"{caption}\"")
        
        # Afficher l'image
        self.display_result(image_path, caption, save_path)
        
        return caption
    
    def demo_multiple_images_grid(self, image_dir, max_length=20, max_images=None, cols=3):
        """
        Génère et affiche des captions pour toutes les images d'un dossier dans une grille
        
        Args:
            image_dir (str): Dossier contenant les images
            max_length (int): Longueur maximale des captions
            max_images (int): Nombre maximum d'images à traiter (None = toutes)
            cols (int): Nombre de colonnes dans la grille
        
        Returns:
            dict: Résultats {image_name: caption}
        """
        print("\n" + "="*70)
        print("GÉNÉRATION DE CAPTIONS MULTIPLES")
        print("="*70)
        
        # Extensions d'images supportées
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Trouver toutes les images JPG
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])
        
        # Limiter le nombre d'images si spécifié
        if max_images is not None:
            image_files = image_files[:max_images]
        
        print(f"\nTrouvé {len(image_files)} images")
        
        if len(image_files) == 0:
            print("Aucune image trouvée!")
            return {}
        
        results = {}
        
        # Calculer le nombre de lignes nécessaires
        rows = (len(image_files) + cols - 1) // cols
        
        # Créer la figure avec subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        # S'assurer que axes est toujours un tableau 2D
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        # Traiter chaque image
        for idx, image_file in enumerate(image_files):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]
            
            image_path = os.path.join(image_dir, image_file)
            
            print(f"\n[{idx+1}/{len(image_files)}] {image_file}")
            
            try:
                # Générer la caption
                caption = self.generate_caption(image_path, max_length=max_length)
                print(f"  Caption: \"{caption}\"")
                
                results[image_file] = caption
                
                # Charger et afficher l'image
                img = Image.open(image_path).convert('RGB')
                ax.imshow(img)
                ax.axis('off')
                
                # Ajouter la caption comme titre
                ax.set_title(caption, fontsize=10, wrap=True, pad=10)
            
            except Exception as e:
                print(f"  Erreur: {e}")
                results[image_file] = f"ERROR: {e}"
                ax.axis('off')
                ax.text(0.5, 0.5, f"Erreur:\n{image_file}", 
                       ha='center', va='center', fontsize=8)
        
        # Cacher les axes vides
        for idx in range(len(image_files), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return results


def main():
    """
    Fonction principale - affiche toutes les images de ImagesTest
    """
    print("="*70)
    print("DÉMO IMAGE CAPTIONING - ImagesTest")
    print("="*70)
    
    # Chemins par défaut
    model_path = 'checkpoints/best_model.pth'
    vocab_path = 'data/vocab.pkl'
    images_dir = 'ImagesTest'
    
    # Vérifier que le dossier existe
    if not os.path.exists(images_dir):
        print(f"ERREUR: Le dossier {images_dir} n'existe pas!")
        return
    
    # Créer la démo
    demo = CaptionDemo(
        model_path=model_path,
        vocab_path=vocab_path,
        encoder_type=CONFIG['encoder_type']
    )
    
    # Générer et afficher toutes les images
    results = demo.demo_multiple_images_grid(
        image_dir=images_dir,
        max_length=20,
        max_images=None,  # Toutes les images
        cols=3  # 3 colonnes
    )
    
    # Afficher un résumé
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print(f"Nombre d'images traitées: {len(results)}")
    for img, cap in results.items():
        print(f"  {img}: {cap}")
    
    print("\n" + "="*70)
    print("DÉMO TERMINÉE !")
    print("="*70)


# ============================================================================
# UTILISATION INTERACTIVE (pour Jupyter ou script Python)
# ============================================================================

def quick_demo(image_path, model_path='checkpoints/best_model.pth', 
               vocab_path='data/vocab.pkl', encoder_type='lite'):
    """
    Fonction rapide pour tester une image
    
    Args:
        image_path (str): Chemin vers l'image
        model_path (str): Chemin vers le modèle
        vocab_path (str): Chemin vers le vocabulaire
        encoder_type (str): Type d'encoder
    
    Returns:
        str: Caption générée
    """
    demo = CaptionDemo(model_path, vocab_path, encoder_type)
    return demo.demo_single_image(image_path)


def quick_demo_batch(images_dir='ImagesTest', model_path='checkpoints/best_model.pth',
                     vocab_path='data/vocab.pkl', encoder_type='lite', max_images=None, cols=3):
    """
    Fonction rapide pour tester toutes les images d'un dossier
    
    Args:
        images_dir (str): Dossier contenant les images
        model_path (str): Chemin vers le modèle
        vocab_path (str): Chemin vers le vocabulaire
        encoder_type (str): Type d'encoder
        max_images (int): Nombre maximum d'images (None = toutes)
        cols (int): Nombre de colonnes dans la grille
    
    Returns:
        dict: Résultats {image_name: caption}
    """
    demo = CaptionDemo(model_path, vocab_path, encoder_type)
    return demo.demo_multiple_images_grid(images_dir, max_images=max_images, cols=cols)


if __name__ == "__main__":
    # Afficher toutes les images de ImagesTest
    main()
    plt.show(block = True)
    
    # ========================================================================
    # EXEMPLES D'UTILISATION
    # ========================================================================
    """
    # Dans un script Python ou Jupyter:
    # ----------------------------------
    
    from demo import quick_demo_batch
    
    # Afficher toutes les images de ImagesTest
    results = quick_demo_batch('ImagesTest')
    
    # Afficher seulement les 9 premières images
    results = quick_demo_batch('ImagesTest', max_images=9, cols=3)
    
    # Afficher avec un modèle différent
    results = quick_demo_batch('ImagesTest', 
                               model_path='checkpoints/another_model.pth',
                               encoder_type='full')
    """