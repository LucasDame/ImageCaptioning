"""
Visualisation de l'attention pour Image Captioning COCO
=========================================================

Compatible avec encoder_type='attention' et encoder_type='densenet'.
La grille spatiale est détectée automatiquement depuis l'encodeur chargé
(attribut grid_size) — pas de changement à faire manuellement.

Fonctionnalités :
  - Filtrage des mots de liaison (stop words) : affichés en opacité
    réduite et fond grisé pour ne pas polluer la grille visuelle.
  - Deux vues : grille individuelle par mot + overlay moyen.
  - La moyenne de l'overlay n'utilise QUE les mots de contenu.
  - Sauvegarde automatique si pas d'affichage interactif.

Utilisation rapide :
    python visualize_attention.py
    python visualize_attention.py --image ImagesTest/dog.jpg
    python visualize_attention.py --image_dir ImagesTest/ --save_dir attention_output/
"""

import torch
import os
import argparse
import numpy as np
from PIL import Image

# ── Backend matplotlib ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
_DISPLAY_MODE = 'save'

for _backend in ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'wxAgg', 'MacOSX']:
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt
        _fig = _plt.figure(); _plt.close(_fig)
        _DISPLAY_MODE = 'interactive'
        break
    except Exception:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt

if _DISPLAY_MODE == 'save':
    print("[INFO] Pas d'affichage interactif → figures sauvegardées.")
# ─────────────────────────────────────────────────────────────────────────────

from torchvision import transforms
from utils.vocabulary import Vocabulary
from models2.caption_model2 import load_model
from config_coco3 import CONFIG


# =============================================================================
# STOP WORDS — mots de liaison sans ancrage visuel
# =============================================================================

STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
    'those', 'it', 'its', 'up', 'as', 'into', 'than', 'some', 'there',
    'their', 'his', 'her', 'our', 'your', 'my', 'about', 'over', 'after',
    'before', 'while', 'through', 'between', 'each', 'no', 'not', 'so',
    'if', 'then', 'very', 'also', 'just', 'out', 'near', 'next', 'two',
    '<end>', '<start>', '<unk>', '<pad>',
})


def is_stop_word(word):
    return word.lower().rstrip('.,!?;:') in STOP_WORDS


# =============================================================================
# VISUALISEUR PRINCIPAL
# =============================================================================

class AttentionVisualizerCOCO:
    """
    Charge un modèle COCO entraîné avec encoder_type='attention' ou 'densenet'
    et génère des visualisations des poids d'attention mot par mot.

    La taille de la grille (grid_size) est détectée automatiquement depuis
    l'encodeur chargé — fonctionne avec 7×7=49 (défaut) ou toute autre valeur.
    """

    _val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, model_path, vocab_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device : {self.device}")

        self.model, info = load_model(model_path, device=self.device)
        self.model.eval()

        # Détecter la taille de grille depuis l'encodeur
        # (fonctionne pour EncoderSpatial et EncoderDenseNet)
        self.grid_size = getattr(self.model.encoder, 'grid_size', 7)
        print(f"Grille spatiale détectée : {self.grid_size}×{self.grid_size} "
              f"= {self.grid_size**2} régions")

        self.vocabulary = info.get('vocab')
        if self.vocabulary is None and vocab_path:
            self.vocabulary = Vocabulary.load(vocab_path)
        if self.vocabulary is None:
            raise ValueError(
                "Vocabulaire introuvable dans le checkpoint. "
                "Passez vocab_path='data/coco_vocab.pkl'."
            )

        self.start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        self.end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]

    # ──────────────────────────────────────────────────────────────────────────

    def _load_image(self, image_path):
        pil = Image.open(image_path).convert('RGB')
        display_t   = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        img_display = display_t(pil)
        img_tensor  = self._val_transform(pil).unsqueeze(0).to(self.device)
        return img_display, img_tensor

    def _alpha_to_heatmap(self, alpha, out_size=224):
        """
        alpha (P,) → heatmap np.ndarray (out_size, out_size) dans [0, 1].
        Upscale bilinéaire depuis grille grid_size×grid_size.
        Utilise self.grid_size détecté automatiquement depuis l'encodeur.
        """
        alpha_np = alpha.cpu().detach().float().numpy()
        grid     = alpha_np.reshape(self.grid_size, self.grid_size)
        pil_hm   = Image.fromarray((grid * 255).astype(np.uint8))
        pil_hm   = pil_hm.resize((out_size, out_size), Image.BILINEAR)
        return np.array(pil_hm) / 255.0

    def _save_or_show(self, fig, save_path=None):
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {save_path}")
        if _DISPLAY_MODE == 'interactive':
            plt.show()
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────

    def plot_attention_grid(self, image_path, tokens, alphas, words,
                            save_path=None, n_cols=5):
        """
        Grille d'images : une case par mot.
        Stop words : opacité réduite + fond grisé + ◌.
        Mots de contenu : heatmap pleine opacité.
        """
        img_display, _ = self._load_image(image_path)
        img_np = np.array(img_display)

        display_items = [
            (w, a) for w, a in zip(words, alphas)
            if w.lower() not in ('<end>', '<start>')
        ]
        if not display_items:
            print("  Aucun mot à afficher.")
            return

        n_words = len(display_items)
        n_rows  = (n_words + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.8, n_rows * 3.0))
        axes = np.array(axes).reshape(n_rows, n_cols)

        caption_str = ' '.join(
            w for w, _ in display_items
            if w.lower() not in ('<pad>', '<unk>')
        )
        fig.suptitle(
            f'{os.path.basename(image_path)}\n"{caption_str}"',
            fontsize=10, fontweight='bold', y=1.01
        )

        for idx, (word, alpha) in enumerate(display_items):
            r, c = divmod(idx, n_cols)
            ax   = axes[r, c]
            hm   = self._alpha_to_heatmap(alpha)
            stop = is_stop_word(word)

            ax.imshow(img_np)
            hm_alpha = 0.20 if stop else 0.50
            ax.imshow(hm, cmap='jet', alpha=hm_alpha,
                      vmin=0, vmax=max(hm.max(), 1e-6))

            label     = word + (' ◌' if stop else '')
            title_col = '#aaaaaa' if stop else 'white'
            bg_col    = '#444444' if stop else '#111111'
            fw        = 'normal'  if stop else 'bold'
            ax.set_title(label, fontsize=9, color=title_col, fontweight=fw,
                         bbox=dict(boxstyle='round,pad=0.15',
                                   facecolor=bg_col, alpha=0.75, linewidth=0))
            ax.axis('off')

        for idx in range(n_words, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].axis('off')

        fig.text(0.01, -0.01,
                 '◌ = mot de liaison — attention diffuse, pas d\'ancrage visuel fiable',
                 fontsize=7, color='gray', style='italic')

        self._save_or_show(fig, save_path)

    # ──────────────────────────────────────────────────────────────────────────

    def plot_attention_overlay(self, image_path, alphas, words,
                               save_path=None):
        """
        Vue synthétique 2 panneaux :
          Gauche  : image originale
          Droite  : heatmap moyenne (mots de contenu uniquement)
        """
        img_display, _ = self._load_image(image_path)
        img_np = np.array(img_display)

        content_alphas = [
            a for w, a in zip(words, alphas)
            if not is_stop_word(w)
            and w.lower() not in ('<end>', '<start>', '<pad>', '<unk>')
        ]
        source     = content_alphas if content_alphas else list(alphas)
        mean_alpha = torch.stack(source, dim=0).mean(dim=0)
        hm         = self._alpha_to_heatmap(mean_alpha)

        caption_str = ' '.join(
            w for w in words
            if w.lower() not in ('<end>', '<start>', '<pad>')
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            f'{os.path.basename(image_path)}\n"{caption_str}"',
            fontsize=10, fontweight='bold'
        )

        axes[0].imshow(img_np)
        axes[0].set_title('Image originale', fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(img_np)
        im = axes[1].imshow(hm, cmap='jet', alpha=0.50,
                             vmin=0, vmax=max(hm.max(), 1e-6))
        axes[1].set_title(
            'Attention moyenne\n(mots de contenu uniquement)', fontsize=9
        )
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        self._save_or_show(fig, save_path)

    # ──────────────────────────────────────────────────────────────────────────

    def visualize_single_image(self, image_path, save_dir=None,
                               method='beam_search', max_length=20):
        """Pipeline complet pour une image : génère grille + overlay."""
        print(f"\n{'='*60}")
        print(f"Image : {image_path}")

        _, img_tensor = self._load_image(image_path)

        with torch.no_grad():
            tokens, alphas = self.model.generate_caption_with_attention(
                img_tensor.squeeze(0),
                max_length=max_length,
                start_token=self.start_token,
                end_token=self.end_token,
                method=method
            )

        words = [self.vocabulary.idx2word.get(t, '<unk>') for t in tokens]
        caption = ' '.join(
            w for w in words
            if w.lower() not in ('<end>', '<start>', '<pad>')
        )
        print(f"Caption    : {caption}")

        sw_count = sum(1 for w in words if is_stop_word(w))
        print(f"Stop words : {sw_count}/{len(words)} (affichés grisés dans la grille)")
        print(f"Grille     : {self.grid_size}×{self.grid_size} régions")

        basename = os.path.splitext(os.path.basename(image_path))[0]
        path_grid = path_overlay = None
        if save_dir:
            path_grid    = os.path.join(save_dir, f'{basename}_attention_grid.png')
            path_overlay = os.path.join(save_dir, f'{basename}_attention_overlay.png')

        self.plot_attention_grid(
            image_path, tokens, alphas, words, save_path=path_grid
        )
        self.plot_attention_overlay(
            image_path, alphas, words, save_path=path_overlay
        )
        return caption, tokens, alphas

    def visualize_multiple_images(self, image_dir, save_dir=None,
                                  max_images=10, method='beam_search'):
        """Applique visualize_single_image sur toutes les images d'un dossier."""
        supported = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [
            f for f in sorted(os.listdir(image_dir))
            if os.path.splitext(f)[1].lower() in supported
        ][:max_images]

        if not images:
            print(f"Aucune image trouvée dans {image_dir}")
            return

        print(f"\n{len(images)} images trouvées dans {image_dir}")
        for fname in images:
            self.visualize_single_image(
                os.path.join(image_dir, fname),
                save_dir=save_dir, method=method
            )


# =============================================================================
# FONCTIONS RAPIDES
# =============================================================================

def quick_attention(image_path,
                    model_path='checkpoints_coco2/best_model.pth',
                    save_dir='attention_output',
                    method='beam_search'):
    viz = AttentionVisualizerCOCO(model_path)
    return viz.visualize_single_image(image_path, save_dir=save_dir,
                                      method=method)


def quick_attention_batch(image_dir='ImagesTest',
                          model_path='checkpoints_coco2/best_model.pth',
                          save_dir='attention_output',
                          max_images=20):
    viz = AttentionVisualizerCOCO(model_path)
    viz.visualize_multiple_images(image_dir, save_dir=save_dir,
                                  max_images=max_images)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention visualizer COCO')
    parser.add_argument('--image',      type=str, default=None)
    parser.add_argument('--image_dir',  type=str, default='ImagesTest')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_coco2/best_model.pth')
    parser.add_argument('--save_dir',   type=str, default='attention_output')
    parser.add_argument('--max_images', type=int, default=20)
    parser.add_argument('--method',     type=str, default='beam_search',
                        choices=['greedy', 'beam_search'])
    args = parser.parse_args()

    viz = AttentionVisualizerCOCO(args.model_path)

    if args.image:
        viz.visualize_single_image(args.image, save_dir=args.save_dir,
                                   method=args.method)
    else:
        viz.visualize_multiple_images(args.image_dir, save_dir=args.save_dir,
                                      max_images=args.max_images,
                                      method=args.method)