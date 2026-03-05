"""
Script de Démo COCO pour Image Captioning
==========================================

Version COCO de demo.py.

Différences avec la version Flickr8k :
  - Imports depuis config_coco (chemins, encoder_type, beam_width)
  - load_model détecte automatiquement l'encoder_type depuis le checkpoint
  - La méthode de génération (greedy / beam_search) et le beam_width
    sont lus depuis CONFIG
  - Le dossier de démo par défaut est 'ImagesTest' (identique)
  - Compatible avec les trois modes : 'lite', 'full', 'attention'
"""

import torch
import os
from PIL import Image

# ── Détection automatique du backend matplotlib ──────────────────────────────
import matplotlib
matplotlib.use('Agg')
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
          "sauvegardées dans le dossier output_captions_coco/")
# ─────────────────────────────────────────────────────────────────────────────

from utils.vocabulary import Vocabulary
from utils.preprocessing_coco import ImagePreprocessor
from models2.caption_model2 import load_model
from config_coco2 import CONFIG


class CaptionDemoCOCO:
    """
    Classe pour faire des démos de génération de captions — dataset COCO.

    Utilise load_model sans préciser encoder_type : le type est détecté
    automatiquement depuis le checkpoint (sauvegardé par save_model).
    """

    def __init__(self, model_path, vocab_path=None):
        """
        Args:
            model_path (str): Chemin vers le checkpoint entraîné sur COCO
            vocab_path (str): Chemin vers le vocabulaire COCO (optionnel
                              si déjà sauvegardé dans le checkpoint)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de : {self.device}")

        print(f"\nChargement du modèle depuis {model_path}...")
        # encoder_type détecté automatiquement depuis le checkpoint
        self.model, info = load_model(model_path, device=self.device)

        self.vocabulary = info['vocab']
        if self.vocabulary is None and vocab_path is not None:
            print(f"Chargement du vocabulaire depuis {vocab_path}...")
            self.vocabulary = Vocabulary.load(vocab_path)
        elif self.vocabulary is None:
            raise ValueError(
                "Vocabulaire non trouvé dans le checkpoint. "
                "Spécifiez vocab_path='data/coco_vocab.pkl'."
            )

        self.image_preprocessor = ImagePreprocessor(
            image_size=CONFIG['image_size'], normalize=True
        )

        self.start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        self.end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]
        self.pad_token   = self.vocabulary.word2idx[self.vocabulary.pad_token]

        self.method     = CONFIG.get('generation_method', 'greedy')
        self.beam_width = CONFIG.get('beam_width', 3)
        self.max_length = CONFIG.get('max_caption_length', 20)

        print(f"✓ Modèle chargé et prêt !")
        print(f"  Méthode de génération : {self.method}"
              + (f" (beam_width={self.beam_width})"
                 if self.method == 'beam_search' else ""))

    # ──────────────────────────────────────────────────────────────────────────

    def generate_caption(self, image_path, max_length=None, method=None):
        """
        Génère une caption pour une image.

        Args:
            image_path (str): Chemin vers l'image
            max_length (int): Longueur maximale (None = valeur de la config)
            method (str): 'greedy' ou 'beam_search' (None = valeur de la config)

        Returns:
            str: Caption générée
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image non trouvée : {image_path}")

        max_length = max_length or self.max_length
        method     = method     or self.method

        image = self.image_preprocessor(image_path, is_training=False)
        image = image.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if method == 'beam_search':
                features = self.model.encoder(image)
                caption_indices = self.model.decoder.generate_beam_search(
                    features,
                    beam_width=self.beam_width,
                    max_length=max_length,
                    start_token=self.start_token,
                    end_token=self.end_token
                )
            else:
                caption_indices = self.model.generate_caption(
                    image,
                    max_length=max_length,
                    start_token=self.start_token,
                    end_token=self.end_token,
                    method='greedy'
                )

        caption_tokens = [
            t.item() if torch.is_tensor(t) else t
            for t in caption_indices[0]
            if (t.item() if torch.is_tensor(t) else t)
            not in [self.start_token, self.end_token, self.pad_token]
        ]

        return self.vocabulary.denumericalize(caption_tokens)

    # ──────────────────────────────────────────────────────────────────────────

    def _show_or_save(self, fig, save_path=None):
        """
        Affiche ou sauvegarde la figure selon le backend disponible.
        """
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {save_path}")
        elif _DISPLAY_MODE == 'save':
            os.makedirs('output_captions_coco', exist_ok=True)
            title     = fig.axes[0].get_title().split('\n')[0]
            safe_name = "".join(
                c if c.isalnum() or c in '-_' else '_' for c in title
            )
            out = os.path.join('output_captions_coco', f"{safe_name}.png")
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {out}")
        else:
            plt.show()

        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────

    def display_result(self, image_path, caption, save_path=None):
        """
        Affiche l'image avec sa caption (ou la sauvegarde).
        """
        img = Image.open(image_path).convert('RGB')

        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(
            f'Caption: "{caption}"',
            fontsize=16, fontweight='bold', pad=20, wrap=True
        )
        plt.tight_layout()

        self._show_or_save(fig, save_path)

    # ──────────────────────────────────────────────────────────────────────────

    def demo_single_image(self, image_path, max_length=None, save_path=None):
        """
        Démo complète pour une seule image.

        Args:
            image_path (str): Chemin vers l'image
            max_length (int): Longueur maximale (None = config)
            save_path (str): Chemin pour sauvegarder (optionnel)

        Returns:
            str: Caption générée
        """
        print("\n" + "="*70)
        print("GÉNÉRATION DE CAPTION (COCO)")
        print("="*70)
        print(f"\nImage : {image_path}")
        print("Génération en cours...")

        caption = self.generate_caption(image_path, max_length=max_length)
        print(f'\nCaption générée : "{caption}"')

        self.display_result(image_path, caption, save_path)
        return caption

    # ──────────────────────────────────────────────────────────────────────────

    def demo_multiple_images(self, image_dir, max_length=None, max_images=None):
        """
        Génère des captions pour toutes les images d'un dossier, une par une.

        - Mode interactif  : une fenêtre s'ouvre pour chaque image.
        - Mode non-interactif : figures sauvegardées dans output_captions_coco/.

        Args:
            image_dir (str): Dossier contenant les images
            max_length (int): Longueur maximale (None = config)
            max_images (int): Nombre max d'images (None = toutes)

        Returns:
            dict: {image_name: caption}
        """
        print("\n" + "="*70)
        print("GÉNÉRATION DE CAPTIONS MULTIPLES (COCO)")
        print("="*70)

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
            print(f"\n[{idx+1}/{len(image_files)}] {image_file}")

            try:
                caption = self.generate_caption(image_path, max_length=max_length)
                print(f'  Caption : "{caption}"')
                results[image_file] = caption

                img = Image.open(image_path).convert('RGB')
                fig = plt.figure(figsize=(8, 7))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    f'[{idx+1}/{len(image_files)}]  {image_file}\n\n"{caption}"',
                    fontsize=14, fontweight='bold', pad=15
                )
                plt.tight_layout()

                self._show_or_save(fig)

            except Exception as e:
                print(f"  Erreur : {e}")
                results[image_file] = f"ERROR: {e}"

        return results


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Fonction principale — affiche toutes les images de ImagesTest
    avec le modèle COCO.
    """
    print("="*70)
    print("DÉMO IMAGE CAPTIONING COCO - ImagesTest")
    print("="*70)

    model_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    vocab_path = CONFIG['vocab_path']
    images_dir = 'ImagesTest'

    if not os.path.exists(model_path):
        print(f"ERREUR : checkpoint introuvable → {model_path}")
        print("Lancez d'abord l'entraînement : python train_coco.py")
        return

    if not os.path.exists(images_dir):
        print(f"ERREUR : dossier {images_dir} introuvable !")
        return

    demo = CaptionDemoCOCO(
        model_path=model_path,
        vocab_path=vocab_path
    )

    results = demo.demo_multiple_images(
        image_dir=images_dir,
        max_images=None
    )

    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print(f"Nombre d'images traitées : {len(results)}")
    for img, cap in results.items():
        print(f"  {img} : {cap}")

    print("\n" + "="*70)
    print("DÉMO COCO TERMINÉE !")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# UTILISATION INTERACTIVE (Jupyter / script Python)
# ─────────────────────────────────────────────────────────────────────────────

def quick_demo(image_path,
               model_path=None,
               vocab_path=None):
    """
    Teste une seule image avec le modèle COCO.

    Args:
        image_path (str): Chemin vers l'image
        model_path (str): Chemin vers le checkpoint (None = config par défaut)
        vocab_path (str): Chemin vers le vocabulaire (None = config par défaut)
    """
    model_path = model_path or os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    vocab_path = vocab_path or CONFIG['vocab_path']
    demo = CaptionDemoCOCO(model_path, vocab_path)
    return demo.demo_single_image(image_path)


def quick_demo_batch(images_dir='ImagesTest',
                     model_path=None,
                     vocab_path=None,
                     max_images=None):
    """
    Teste toutes les images d'un dossier avec le modèle COCO.

    Args:
        images_dir (str): Dossier contenant les images
        model_path (str): Chemin vers le checkpoint (None = config par défaut)
        vocab_path (str): Chemin vers le vocabulaire (None = config par défaut)
        max_images (int): Nombre max d'images (None = toutes)
    """
    model_path = model_path or os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    vocab_path = vocab_path or CONFIG['vocab_path']
    demo = CaptionDemoCOCO(model_path, vocab_path)
    return demo.demo_multiple_images(images_dir, max_images=max_images)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

    # ── Exemples d'utilisation ────────────────────────────────────────────────
    # from demo_coco import quick_demo_batch
    #
    # # Toutes les images
    # results = quick_demo_batch('ImagesTest')
    #
    # # Seulement les 9 premières
    # results = quick_demo_batch('ImagesTest', max_images=9)
    #
    # # Avec un checkpoint spécifique
    # results = quick_demo_batch('ImagesTest',
    #                            model_path='checkpoints_coco/checkpoint_epoch_10.pth')
