"""
Visualisation des cartes d'attention — Image Captioning COCO
=============================================================

Version attention de demo_coco2.py.
Pour chaque image, génère deux figures :
  1. Grille mot par mot : l'image avec la heatmap d'attention superposée
     pour chaque mot de la caption.
  2. Overlay global : l'image originale + la moyenne de toutes les alphas.

Requiert encoder_type='attention' (EncoderSpatial + DecoderWithAttention).

Utilisation :
    python visualize_attention.py               # traite ImagesTest/
    python visualize_attention.py --image img.jpg
"""

import os
import math

import torch
import numpy as np
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
          "sauvegardées dans le dossier output_attention/")
# ─────────────────────────────────────────────────────────────────────────────

from utils.vocabulary import Vocabulary
from utils.preprocessing_coco import ImagePreprocessor
from models2.caption_model2 import load_model
from config_coco2 import CONFIG


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class AttentionVisualizerCOCO:
    """
    Visualise les cartes d'attention du modèle COCO avec encoder_type='attention'.

    Génère pour chaque image :
      - Une grille de sous-figures (une par mot) avec la heatmap d'attention
      - Un overlay global montrant l'attention moyenne sur toute la caption
    """

    def __init__(self, model_path, vocab_path=None):
        """
        Args:
            model_path (str): Chemin vers le checkpoint entraîné
            vocab_path (str): Chemin vers le vocabulaire (optionnel si dans le checkpoint)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de : {self.device}")

        print(f"\nChargement du modèle depuis {model_path}...")
        self.model, info = load_model(model_path, device=self.device,
                                      encoder_type='attention')

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

        self.method     = CONFIG.get('generation_method', 'beam_search')
        self.beam_width = CONFIG.get('beam_width', 5)
        self.max_length = CONFIG.get('max_caption_length', 20)
        self.grid_size  = 7   # EncoderSpatial produit une grille 7×7 = 49 régions

        print(f"✓ Modèle chargé et prêt !")
        print(f"  Méthode de génération : {self.method}"
              + (f" (beam_width={self.beam_width})"
                 if self.method == 'beam_search' else ""))

    # ──────────────────────────────────────────────────────────────────────────

    def _load_image_for_display(self, image_path):
        """
        Charge l'image en PIL avec le même crop que val_transform (resize 256,
        center crop 224), pour que la heatmap soit parfaitement alignée.
        """
        size = CONFIG['image_size']
        img  = Image.open(image_path).convert('RGB')
        w, h = img.size
        scale = 256 / min(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        left  = (img.width  - size) // 2
        top   = (img.height - size) // 2
        img   = img.crop((left, top, left + size, top + size))
        return img

    def _generate_with_attention(self, image_path):
        """
        Prétraite l'image et génère la caption avec les poids d'attention.

        Returns:
            tokens : list[int]
            alphas : Tensor (seq_len, num_pixels)
            caption: str
        """
        image = self.image_preprocessor(image_path, is_training=False)
        image = image.unsqueeze(0).to(self.device)

        tokens, alphas = self.model.generate_caption_with_attention(
            image,
            max_length=self.max_length,
            start_token=self.start_token,
            end_token=self.end_token,
            method=self.method
        )

        caption_tokens = [
            t for t in tokens
            if t not in [self.start_token, self.end_token, self.pad_token]
        ]
        caption = self.vocabulary.denumericalize(caption_tokens)
        return tokens, alphas, caption

    def _alpha_to_heatmap(self, alpha, display_size):
        """
        Convertit un vecteur alpha (num_pixels,) en np.array (H, W) [0,1]
        upscalé à display_size par interpolation bicubique.
        """
        att_map = alpha.cpu().numpy().reshape(self.grid_size, self.grid_size)
        att_img = Image.fromarray(
            (att_map * 255).astype(np.uint8)
        ).resize(display_size, resample=Image.BICUBIC)
        return np.array(att_img, dtype=np.float32) / 255.0

    # ──────────────────────────────────────────────────────────────────────────

    def _show_or_save(self, fig, save_path=None):
        """
        Affiche ou sauvegarde la figure selon le backend disponible.
        Même logique que dans demo_coco2.py.
        """
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {save_path}")
        elif _DISPLAY_MODE == 'save':
            os.makedirs('output_attention', exist_ok=True)
            suptitle  = fig._suptitle.get_text() if fig._suptitle else 'attention'
            safe_name = "".join(
                c if c.isalnum() or c in '-_' else '_' for c in suptitle
            )[:60]
            out = os.path.join('output_attention', f"{safe_name}.png")
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {out}")
        else:
            plt.show()

        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────

    def plot_attention_grid(self, image_path, tokens, alphas, caption,
                            save_path=None):
        """
        Grille de sous-figures : une par mot.
        Chaque sous-figure = image + heatmap d'attention superposée.
        """
        special = {self.start_token, self.end_token, self.pad_token}

        pairs = [
            (self.vocabulary.idx2word.get(idx, '<UNK>'), alpha)
            for idx, alpha in zip(tokens, alphas)
            if idx not in special
        ]

        if not pairs:
            print("  ⚠️  Aucun mot à visualiser.")
            return

        img_pil   = self._load_image_for_display(image_path)
        img_array = np.array(img_pil)
        W, H      = img_pil.size

        n_words = len(pairs)
        n_cols  = min(5, n_words)
        n_rows  = math.ceil(n_words / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3.5 * n_rows + 0.8)
        )
        fig.suptitle(
            f'"{caption}"',
            fontsize=12, fontstyle='italic', fontweight='bold', y=1.01
        )

        # Normaliser axes en liste 2D pour indexation uniforme
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [list(axes)]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = [list(row) for row in axes]

        for i, (word, alpha) in enumerate(pairs):
            row, col = divmod(i, n_cols)
            ax       = axes[row][col]

            heatmap = self._alpha_to_heatmap(alpha, (W, H))

            ax.imshow(img_array)
            ax.imshow(heatmap, cmap='jet', alpha=0.45,
                      vmin=0, vmax=heatmap.max())
            ax.set_title(word, fontsize=11, fontweight='bold', pad=4)
            ax.axis('off')

        # Masquer les axes vides (dernière ligne incomplète)
        for i in range(n_words, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row][col].axis('off')

        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ──────────────────────────────────────────────────────────────────────────

    def plot_attention_overlay(self, image_path, tokens, alphas, caption,
                               save_path=None):
        """
        Vue globale : image originale | attention moyenne sur toute la caption.
        """
        special = {self.start_token, self.end_token, self.pad_token}

        valid_alphas = [
            alpha for idx, alpha in zip(tokens, alphas)
            if idx not in special
        ]

        if not valid_alphas:
            return

        img_pil   = self._load_image_for_display(image_path)
        img_array = np.array(img_pil)
        W, H      = img_pil.size

        mean_alpha = torch.stack(valid_alphas).mean(dim=0)
        heatmap    = self._alpha_to_heatmap(mean_alpha, (W, H))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f'"{caption}"',
            fontsize=13, fontstyle='italic', fontweight='bold', y=1.02
        )

        ax1.imshow(img_array)
        ax1.set_title('Image originale', fontsize=11)
        ax1.axis('off')

        ax2.imshow(img_array)
        im = ax2.imshow(heatmap, cmap='jet', alpha=0.5)
        ax2.set_title('Attention globale (moyenne)', fontsize=11)
        ax2.axis('off')

        cbar = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.04)
        cbar.set_label("Intensité d'attention", fontsize=9)

        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ──────────────────────────────────────────────────────────────────────────

    def visualize_single_image(self, image_path, save_dir=None):
        """
        Visualise l'attention pour une seule image.
        Génère les deux figures (grille + overlay).

        Args:
            image_path (str): Chemin vers l'image
            save_dir   (str): Dossier de sauvegarde (None = affichage ou output_attention/)

        Returns:
            str: Caption générée
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image non trouvée : {image_path}")

        print(f"\nImage : {image_path}")
        print("Génération en cours...")

        tokens, alphas, caption = self._generate_with_attention(image_path)
        print(f'  Caption : "{caption}"')
        print(f"  Tokens  : {len(tokens)} mots  |  Alphas : {alphas.shape}")

        basename = os.path.splitext(os.path.basename(image_path))[0]

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path_grid    = os.path.join(save_dir, f"{basename}_attention_grid.png")
            path_overlay = os.path.join(save_dir, f"{basename}_attention_overlay.png")
        else:
            path_grid    = None
            path_overlay = None

        self.plot_attention_grid(
            image_path, tokens, alphas, caption, save_path=path_grid
        )
        self.plot_attention_overlay(
            image_path, tokens, alphas, caption, save_path=path_overlay
        )

        return caption

    # ──────────────────────────────────────────────────────────────────────────

    def visualize_multiple_images(self, image_dir, save_dir=None,
                                  max_images=None):
        """
        Visualise l'attention pour toutes les images d'un dossier.

        Args:
            image_dir  (str): Dossier contenant les images
            save_dir   (str): Dossier de sauvegarde (None = output_attention/)
            max_images (int): Nombre max d'images (None = toutes)

        Returns:
            dict: {image_name: caption}
        """
        print("\n" + "="*70)
        print("VISUALISATION ATTENTION — IMAGES MULTIPLES")
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

        results  = {}
        out_dir  = save_dir or 'output_attention'

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            print(f"\n[{idx+1}/{len(image_files)}] {image_file}")

            try:
                caption = self.visualize_single_image(
                    image_path, save_dir=out_dir
                )
                results[image_file] = caption

            except Exception as e:
                print(f"  Erreur : {e}")
                results[image_file] = f"ERROR: {e}"

        return results


# =============================================================================
# FONCTIONS RAPIDES
# =============================================================================

def quick_attention(image_path, model_path=None, vocab_path=None,
                    save_dir=None):
    """
    Visualise l'attention pour une seule image.

    Args:
        image_path (str): Chemin vers l'image
        model_path (str): Checkpoint (None = config par défaut)
        vocab_path (str): Vocabulaire (None = config par défaut)
        save_dir   (str): Dossier de sauvegarde (None = affichage)
    """
    model_path = model_path or os.path.join(
        CONFIG['checkpoint_dir'], 'best_model.pth'
    )
    vocab_path = vocab_path or CONFIG['vocab_path']
    viz = AttentionVisualizerCOCO(model_path, vocab_path)
    return viz.visualize_single_image(image_path, save_dir=save_dir)


def quick_attention_batch(images_dir='ImagesTest', model_path=None,
                          vocab_path=None, save_dir=None, max_images=None):
    """
    Visualise l'attention pour toutes les images d'un dossier.

    Args:
        images_dir (str): Dossier d'images (défaut : ImagesTest)
        model_path (str): Checkpoint (None = config par défaut)
        vocab_path (str): Vocabulaire (None = config par défaut)
        save_dir   (str): Dossier de sauvegarde (None = output_attention/)
        max_images (int): Nombre max d'images (None = toutes)
    """
    model_path = model_path or os.path.join(
        CONFIG['checkpoint_dir'], 'best_model.pth'
    )
    vocab_path = vocab_path or CONFIG['vocab_path']
    viz = AttentionVisualizerCOCO(model_path, vocab_path)
    return viz.visualize_multiple_images(
        images_dir, save_dir=save_dir, max_images=max_images
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualise les cartes d'attention du modèle COCO."
    )
    parser.add_argument('--image',      default=None,
                        help='Image unique à analyser (optionnel)')
    parser.add_argument('--images_dir', default='ImagesTest',
                        help="Dossier d'images en mode batch (défaut : ImagesTest)")
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint .pth (défaut : config)')
    parser.add_argument('--vocab',      default=None,
                        help='Vocabulaire .pkl (défaut : config)')
    parser.add_argument('--save_dir',   default=None,
                        help='Dossier de sauvegarde (défaut : output_attention/)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Nombre max d\'images en mode batch')
    args = parser.parse_args()

    print("="*70)
    print("VISUALISATION ATTENTION COCO")
    print("="*70)

    model_path = args.checkpoint or os.path.join(
        CONFIG['checkpoint_dir'], 'best_model.pth'
    )
    vocab_path = args.vocab or CONFIG['vocab_path']

    if not os.path.exists(model_path):
        print(f"ERREUR : checkpoint introuvable → {model_path}")
        print("Lancez d'abord l'entraînement : python train_coco2.py")
        return

    viz = AttentionVisualizerCOCO(model_path, vocab_path)

    if args.image:
        viz.visualize_single_image(args.image, save_dir=args.save_dir)
    else:
        if not os.path.exists(args.images_dir):
            print(f"ERREUR : dossier {args.images_dir} introuvable !")
            return

        results = viz.visualize_multiple_images(
            image_dir=args.images_dir,
            save_dir=args.save_dir,
            max_images=args.max_images
        )

        print("\n" + "="*70)
        print("RÉSUMÉ")
        print("="*70)
        print(f"Nombre d'images traitées : {len(results)}")
        for img, cap in results.items():
            print(f"  {img} : {cap}")

    print("\n" + "="*70)
    print("VISUALISATION TERMINÉE !")
    print("="*70)


if __name__ == '__main__':
    main()

    # ── Exemples d'utilisation ─────────────────────────────────────────────────
    # from visualize_attention import quick_attention, quick_attention_batch
    #
    # # Une seule image
    # quick_attention('ImagesTest/photo.jpg')
    #
    # # Toutes les images, sauvegardées dans un dossier spécifique
    # quick_attention_batch('ImagesTest', save_dir='results_attention/')
    #
    # # Avec un checkpoint spécifique, 5 images max
    # quick_attention_batch('ImagesTest',
    #                       model_path='checkpoints_coco2/checkpoint_epoch_10.pth',
    #                       max_images=5)