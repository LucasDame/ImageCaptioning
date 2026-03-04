"""
Script de Démo pour Image Captioning
=====================================

Génère une caption pour une image fournie.
Parfait pour la présentation finale !
"""

import torch
import argparse
from PIL import Image
import os

# ── Détection automatique du backend matplotlib ──────────────────────────────
# Si un écran est disponible (TkAgg, Qt5, etc.) on affiche les fenêtres.
# Sinon on bascule en mode "save" : chaque figure est sauvegardée dans
# output_captions/ et aucune fenêtre n'est ouverte.

import matplotlib
matplotlib.use('Agg')  # backend par défaut (non-interactif)
_DISPLAY_MODE = 'save'

for _backend in ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'wxAgg', 'MacOSX']:
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt
        fig = _plt.figure()
        _plt.close(fig)
        _DISPLAY_MODE = 'interactive'
        break
    except Exception:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt

if _DISPLAY_MODE == 'save':
    print("[INFO] Pas d'affichage interactif détecté → les figures seront "
          "sauvegardées dans le dossier output_captions/")
# ─────────────────────────────────────────────────────────────────────────────

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de: {self.device}")

        print(f"\nChargement du modèle depuis {model_path}...")
        self.model, info = load_model(model_path, device=self.device, encoder_type=encoder_type)

        self.vocabulary = info['vocab']
        if self.vocabulary is None and vocab_path is not None:
            print(f"Chargement du vocabulaire depuis {vocab_path}...")
            self.vocabulary = Vocabulary.load(vocab_path)
        elif self.vocabulary is None:
            raise ValueError("Vocabulaire non trouvé. Spécifiez vocab_path.")

        self.image_preprocessor = ImagePreprocessor(image_size=CONFIG['image_size'], normalize=True)

        self.start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        self.end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]
        self.pad_token   = self.vocabulary.word2idx[self.vocabulary.pad_token]

        print("✓ Modèle chargé et prêt !")

    # ──────────────────────────────────────────────────────────────────────────

    def generate_caption(self, image_path, max_length=20, method='greedy'):
        """
        Génère une caption pour une image.

        Args:
            image_path (str): Chemin vers l'image
            max_length (int): Longueur maximale de la caption
            method (str): 'greedy' ou 'beam_search'

        Returns:
            str: Caption générée
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image non trouvée: {image_path}")

        image = self.image_preprocessor(image_path, is_training=False)
        image = image.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            caption_indices = self.model.generate_caption(
                image,
                max_length=max_length,
                start_token=self.start_token,
                end_token=self.end_token,
                method=method
            )

        caption_tokens = []
        for token in caption_indices[0]:
            token_val = token.item() if torch.is_tensor(token) else token
            if token_val not in [self.start_token, self.end_token, self.pad_token]:
                caption_tokens.append(token_val)

        return self.vocabulary.denumericalize(caption_tokens)

    # ──────────────────────────────────────────────────────────────────────────

    def _show_or_save(self, fig, save_path=None):
        """
        Affiche la figure si un backend interactif est disponible,
        sinon la sauvegarde (dans save_path ou dans output_captions/).

        Args:
            fig: figure matplotlib
            save_path (str): chemin de sauvegarde explicite (optionnel)
        """
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {save_path}")
        elif _DISPLAY_MODE == 'save':
            os.makedirs('output_captions', exist_ok=True)
            # Utilise le titre de la figure comme nom de fichier
            title = fig.axes[0].get_title().split('\n')[0]
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in title)
            out = os.path.join('output_captions', f"{safe_name}.png")
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {out}")
        else:
            plt.show()

        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────

    def display_result(self, image_path, caption, save_path=None):
        """
        Affiche l'image avec sa caption (ou la sauvegarde en mode non-interactif).

        Args:
            image_path (str): Chemin vers l'image
            caption (str): Caption générée
            save_path (str): Chemin pour sauvegarder (optionnel)
        """
        img = Image.open(image_path).convert('RGB')

        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Caption: "{caption}"',
                  fontsize=16,
                  fontweight='bold',
                  pad=20,
                  wrap=True)
        plt.tight_layout()

        self._show_or_save(fig, save_path)

    # ──────────────────────────────────────────────────────────────────────────

    def demo_single_image(self, image_path, max_length=20, save_path=None):
        """
        Démo complète pour une seule image.

        Args:
            image_path (str): Chemin vers l'image
            max_length (int): Longueur maximale de la caption
            save_path (str): Chemin pour sauvegarder (optionnel)

        Returns:
            str: Caption générée
        """
        print("\n" + "=" * 70)
        print("GÉNÉRATION DE CAPTION")
        print("=" * 70)
        print(f"\nImage: {image_path}")
        print("\nGénération en cours...")

        caption = self.generate_caption(image_path, max_length=max_length)
        print(f'\nCaption générée: "{caption}"')

        self.display_result(image_path, caption, save_path)
        return caption

    # ──────────────────────────────────────────────────────────────────────────

    def demo_multiple_images(self, image_dir, max_length=20, max_images=None):
        """
        Génère et affiche des captions pour toutes les images d'un dossier,
        une par une.
        - Mode interactif : une fenêtre s'ouvre pour chaque image (fermer pour continuer).
        - Mode non-interactif : les figures sont sauvegardées dans output_captions/.

        Args:
            image_dir (str): Dossier contenant les images
            max_length (int): Longueur maximale des captions
            max_images (int): Nombre maximum d'images à traiter (None = toutes)

        Returns:
            dict: Résultats {image_name: caption}
        """
        print("\n" + "=" * 70)
        print("GÉNÉRATION DE CAPTIONS MULTIPLES")
        print("=" * 70)

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if max_images is not None:
            image_files = image_files[:max_images]

        print(f"\nTrouvé {len(image_files)} image(s)")

        if not image_files:
            print("Aucune image trouvée !")
            return {}

        results = {}

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            print(f"\n[{idx + 1}/{len(image_files)}] {image_file}")

            try:
                caption = self.generate_caption(image_path, max_length=max_length)
                print(f'  Caption: "{caption}"')
                results[image_file] = caption

                img = Image.open(image_path).convert('RGB')

                fig = plt.figure(figsize=(8, 7))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    f'[{idx + 1}/{len(image_files)}]  {image_file}\n\n"{caption}"',
                    fontsize=14,
                    fontweight='bold',
                    pad=15
                )
                plt.tight_layout()

                self._show_or_save(fig)

            except Exception as e:
                print(f"  Erreur: {e}")
                results[image_file] = f"ERROR: {e}"

        return results


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Fonction principale – affiche toutes les images de ImagesTest.
    """
    print("=" * 70)
    print("DÉMO IMAGE CAPTIONING - ImagesTest")
    print("=" * 70)

    model_path = 'checkpoints/best_model.pth'
    vocab_path  = 'data/vocab.pkl'
    images_dir  = 'ImagesTest'

    if not os.path.exists(images_dir):
        print(f"ERREUR: Le dossier {images_dir} n'existe pas !")
        return

    demo = CaptionDemo(
        model_path=model_path,
        vocab_path=vocab_path,
        encoder_type=CONFIG['encoder_type']
    )

    results = demo.demo_multiple_images(
        image_dir=images_dir,
        max_length=20,
        max_images=None
    )

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Nombre d'images traitées : {len(results)}")
    for img, cap in results.items():
        print(f"  {img}: {cap}")

    print("\n" + "=" * 70)
    print("DÉMO TERMINÉE !")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# UTILISATION INTERACTIVE (Jupyter / script Python)
# ─────────────────────────────────────────────────────────────────────────────

def quick_demo(image_path, model_path='checkpoints/best_model.pth',
               vocab_path='data/vocab.pkl', encoder_type='lite'):
    """Teste une seule image et affiche (ou sauvegarde) le résultat."""
    demo = CaptionDemo(model_path, vocab_path, encoder_type)
    return demo.demo_single_image(image_path)


def quick_demo_batch(images_dir='ImagesTest', model_path='checkpoints/best_model.pth',
                     vocab_path='data/vocab.pkl', encoder_type='lite', max_images=None):
    """Teste toutes les images d'un dossier, une par une."""
    demo = CaptionDemo(model_path, vocab_path, encoder_type)
    return demo.demo_multiple_images(images_dir, max_images=max_images)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

    # ── Exemples d'utilisation ────────────────────────────────────────────────
    # from demo import quick_demo_batch
    #
    # # Toutes les images
    # results = quick_demo_batch('ImagesTest')
    #
    # # Seulement les 9 premières
    # results = quick_demo_batch('ImagesTest', max_images=9)
    #
    # # Avec un autre modèle
    # results = quick_demo_batch('ImagesTest',
    #                            model_path='checkpoints/another_model.pth',
    #                            encoder_type='full')