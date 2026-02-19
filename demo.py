"""
Script de Démo pour Image Captioning
=====================================

Génère une caption pour une image fournie.
Parfait pour la présentation finale !
"""

import torch
import argparse
from PIL import Image
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
        
        plt.show()
    
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
    
    def demo_multiple_images(self, image_dir, output_dir=None, max_length=20):
        """
        Génère des captions pour toutes les images d'un dossier
        
        Args:
            image_dir (str): Dossier contenant les images
            output_dir (str): Dossier pour sauvegarder les résultats
            max_length (int): Longueur maximale des captions
        
        Returns:
            dict: Résultats {image_name: caption}
        """
        print("\n" + "="*70)
        print("GÉNÉRATION DE CAPTIONS MULTIPLES")
        print("="*70)
        
        # Créer le dossier de sortie si nécessaire
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extensions d'images supportées
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Trouver toutes les images
        image_files = [
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        print(f"\nTrouvé {len(image_files)} images")
        
        results = {}
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            
            print(f"\n[{i}/{len(image_files)}] {image_file}")
            
            try:
                # Générer la caption
                caption = self.generate_caption(image_path, max_length=max_length)
                print(f"  Caption: \"{caption}\"")
                
                results[image_file] = caption
                
                # Sauvegarder si demandé
                if output_dir:
                    save_path = os.path.join(output_dir, f"result_{image_file}")
                    self.display_result(image_path, caption, save_path)
                    plt.close()  # Fermer pour économiser la mémoire
            
            except Exception as e:
                print(f"  Erreur: {e}")
                results[image_file] = f"ERROR: {e}"
        
        return results


def main():
    """
    Fonction principale avec arguments en ligne de commande
    """
    parser = argparse.ArgumentParser(description='Démo d\'Image Captioning')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Chemin vers l\'image (fichier ou dossier)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Chemin vers le modèle entraîné')
    parser.add_argument('--vocab', type=str, default='data/vocab.pkl',
                       help='Chemin vers le vocabulaire')
    parser.add_argument('--encoder', type=str, default='lite', choices=['full', 'lite'],
                       help='Type d\'encoder utilisé')
    parser.add_argument('--max_length', type=int, default=20,
                       help='Longueur maximale de la caption')
    parser.add_argument('--output', type=str, default=None,
                       help='Chemin pour sauvegarder le résultat')
    parser.add_argument('--batch', action='store_true',
                       help='Traiter toutes les images d\'un dossier')
    
    args = parser.parse_args()
    
    # Créer la démo
    demo = CaptionDemo(
        model_path=args.model,
        vocab_path=args.vocab,
        encoder_type=args.encoder
    )
    
    # Mode batch ou single
    if args.batch:
        # Traiter un dossier
        results = demo.demo_multiple_images(
            image_dir=args.image,
            output_dir=args.output,
            max_length=args.max_length
        )
        
        # Afficher un résumé
        print("\n" + "="*70)
        print("RÉSUMÉ")
        print("="*70)
        for img, cap in results.items():
            print(f"{img}: {cap}")
    
    else:
        # Traiter une seule image
        demo.demo_single_image(
            image_path=args.image,
            max_length=args.max_length,
            save_path=args.output
        )
    
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


if __name__ == "__main__":
    # Si exécuté depuis la ligne de commande
    main()
    
    # ========================================================================
    # EXEMPLES D'UTILISATION
    # ========================================================================
    """
    # Ligne de commande:
    # ------------------
    
    # Image unique
    python demo.py --image data/test_image.jpg --model checkpoints/best_model.pth
    
    # Image unique avec sauvegarde
    python demo.py --image data/test_image.jpg --output results/demo_result.png
    
    # Batch de plusieurs images
    python demo.py --image data/test_images/ --batch --output results/batch_results/
    
    
    # Dans un script Python:
    # ----------------------
    
    from demo import quick_demo
    
    caption = quick_demo('data/test_image.jpg')
    print(caption)
    
    
    # Utilisation avancée:
    # --------------------
    
    from demo import CaptionDemo
    
    demo = CaptionDemo('checkpoints/best_model.pth', 'data/vocab.pkl')
    
    # Générer une caption
    caption = demo.generate_caption('image.jpg')
    
    # Afficher le résultat
    demo.display_result('image.jpg', caption, save_path='result.png')
    """