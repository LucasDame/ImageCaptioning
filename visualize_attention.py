"""
visualize_attention.py — Visualisation de l'attention
======================================================

Disponible uniquement avec model='resnet' ou model='densenet'.
Le type est détecté automatiquement depuis le checkpoint.

Source d'images (priorité décroissante) :
    1. --image      : une seule image (chemin explicite)
    2. --image_dir  : dossier explicite
    3. (auto)       : data/coco/test2017/ si le dossier existe
    4. (auto)       : ImagesTest/ sinon

Utilisation :
    # Auto (test2017 si présent, sinon ImagesTest/)
    python visualize_attention.py --model densenet

    # Dossier explicite
    python visualize_attention.py --model resnet --image_dir ImagesTest/

    # Image unique
    python visualize_attention.py --model densenet --image ImagesTest/dog.jpg

    # Forcer test2017
    python visualize_attention.py --model densenet --use_coco_test

    # Méthode de génération
    python visualize_attention.py --model densenet --method greedy
"""

import argparse
import os

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
import matplotlib.cm as cm

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import get_config
from models.caption_model import load_model
from utils.vocabulary import Vocabulary
from utils.preprocessing import ImagePreprocessor


# Dossiers de test — par ordre de priorité (auto-détection)
COCO_TEST_DIR     = os.path.join('data', 'coco', 'test2017')
FALLBACK_TEST_DIR = 'ImagesTest'


def resolve_image_source(args):
    """
    Détermine la source d'images à partir des arguments CLI.

    Priorité :
      1. args.image     → image unique (chemin explicite)
      2. args.image_dir → dossier explicite
      3. data/coco/test2017/  si le dossier existe et contient des images
      4. ImagesTest/          sinon (fallback)

    Returns:
        tuple (mode, path)
          mode = 'single'  → path est le chemin d'une image
          mode = 'folder'  → path est le chemin d'un dossier
    """
    # 1. Image unique explicite
    if args.image:
        return 'single', args.image

    # 2. Dossier explicite
    if args.image_dir:
        return 'folder', args.image_dir

    # 3. Auto-détection : test2017 COCO
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    if os.path.isdir(COCO_TEST_DIR):
        images = [f for f in os.listdir(COCO_TEST_DIR)
                  if os.path.splitext(f)[1].lower() in valid_ext]
        if images:
            print(f"[Auto] Source : {COCO_TEST_DIR}  ({len(images)} images)")
            return 'folder', COCO_TEST_DIR

    # 4. Fallback : ImagesTest/
    if os.path.isdir(FALLBACK_TEST_DIR):
        images = [f for f in os.listdir(FALLBACK_TEST_DIR)
                  if os.path.splitext(f)[1].lower() in valid_ext]
        if images:
            print(f"[Auto] Source : {FALLBACK_TEST_DIR}  ({len(images)} images)")
            return 'folder', FALLBACK_TEST_DIR

    return None, None


# =============================================================================
# STOP WORDS
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
# VISUALISEUR
# =============================================================================

class AttentionVisualizer:
    """
    Visualise les cartes d'attention Bahdanau sur l'image.
    Compatible avec model='resnet' et model='densenet'.
    """

    def __init__(self, checkpoint_path, vocab_path=None, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Device : {self.device}")

        print(f"Chargement du modèle depuis {checkpoint_path}...")
        self.model, info = load_model(checkpoint_path, device=self.device)

        # Vérifier que le modèle supporte l'attention
        if not hasattr(self.model.decoder, 'generate_with_attention'):
            raise ValueError(
                "Ce modèle ne supporte pas la visualisation d'attention.\n"
                "Utilisez --model resnet ou --model densenet."
            )

        self.vocabulary = info.get('vocab')
        if self.vocabulary is None:
            if vocab_path is None:
                raise ValueError(
                    "Vocabulaire absent du checkpoint. "
                    "Spécifiez --vocab_path data/coco_vocab.pkl"
                )
            self.vocabulary = Vocabulary.load(vocab_path)

        # Détecter la taille de la grille depuis l'encodeur
        self.grid_size = getattr(self.model.encoder, 'grid_size', 7)
        print(f"Grille spatiale : {self.grid_size}×{self.grid_size} = {self.grid_size**2} régions")

        # Transform val
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.image_prep = ImagePreprocessor(
            image_size=224, normalize=False,
            val_transform=self.val_transform
        )

        self.start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        self.end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]
        self.pad_token   = self.vocabulary.word2idx[self.vocabulary.pad_token]

        print("✓ Visualiseur prêt !")

    # ──────────────────────────────────────────────────────────────────────────

    def get_caption_and_attention(self, image_path, method='beam_search',
                                  beam_width=5, max_length=20):
        """
        Génère une caption ET récupère les cartes d'attention.

        Returns:
            tokens (list[str]) : mots de la caption générée
            alphas (Tensor)    : poids d'attention (T, num_pixels)
        """
        image_tensor = self.image_prep(image_path, is_training=False)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            tokens, alphas = self.model.generate_caption_with_attention(
                image_tensor,
                max_length=max_length,
                start_token=self.start_token,
                end_token=self.end_token,
                method=method
            )

        # Convertir tokens (indices) en mots
        words = []
        for t in tokens:
            idx = t if isinstance(t, int) else t.item()
            if idx == self.end_token:
                break
            if idx not in [self.start_token, self.pad_token]:
                words.append(self.vocabulary.idx2word.get(idx, '<unk>'))

        return words, alphas

    # ──────────────────────────────────────────────────────────────────────────

    def _load_image_display(self, image_path):
        """Charge l'image pour affichage (Resize+CenterCrop 224, sans normalisation)."""
        display_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        pil = Image.open(image_path).convert('RGB')
        return display_transform(pil)

    def _alpha_to_heatmap(self, alpha, out_size=224):
        """
        alpha (P,) tensor ou ndarray → heatmap np.ndarray (out_size, out_size) dans [0, 1].
        Upscale bilinéaire depuis grille grid_size×grid_size.
        """
        if isinstance(alpha, torch.Tensor):
            alpha_np = alpha.cpu().detach().float().numpy()
        else:
            alpha_np = alpha.astype(np.float32)
        grid   = alpha_np.reshape(self.grid_size, self.grid_size)
        pil_hm = Image.fromarray((grid * 255).astype(np.uint8))
        pil_hm = pil_hm.resize((out_size, out_size), Image.BILINEAR)
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

    def plot_attention_grid(self, image_path, words, alphas,
                            save_path=None, n_cols=5, show_stop_words=False):
        """
        Grille d'images : une case par mot.
        Stop words : opacité réduite + fond grisé + ◌.
        Mots de contenu : heatmap pleine opacité.
        """
        img_display = self._load_image_display(image_path)
        img_np = np.array(img_display)

        display_items = [
            (w, a) for w, a in zip(words, alphas)
            if w.lower() not in ('<end>', '<start>')
            and (show_stop_words or not is_stop_word(w))
        ]
        # Fallback : si tout est filtré, tout afficher
        if not display_items:
            display_items = [
                (w, a) for w, a in zip(words, alphas)
                if w.lower() not in ('<end>', '<start>')
            ]
        if not display_items:
            print("  Aucun mot à afficher.")
            return

        n_words = len(display_items)
        n_cols  = min(n_cols, n_words)
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

    def plot_attention_overlay(self, image_path, words, alphas,
                               save_path=None):
        """
        Vue synthétique 2 panneaux :
          Gauche  : image originale
          Droite  : heatmap moyenne (mots de contenu uniquement)
        """
        img_display = self._load_image_display(image_path)
        img_np = np.array(img_display)

        content_alphas = [
            a for w, a in zip(words, alphas)
            if not is_stop_word(w)
            and w.lower() not in ('<end>', '<start>', '<pad>', '<unk>')
        ]
        source = content_alphas if content_alphas else list(alphas)

        # Empiler et moyenner (supporte tensors et ndarrays)
        if isinstance(source[0], torch.Tensor):
            mean_alpha = torch.stack(source, dim=0).mean(dim=0)
        else:
            mean_alpha = np.mean(source, axis=0)
        hm = self._alpha_to_heatmap(mean_alpha)

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

    def visualize(self, image_path, method='beam_search', beam_width=5,
                  max_length=20, save_dir=None, show_stop_words=False):
        """
        Pipeline complet pour une image :
          1. Génère la caption + les poids d'attention
          2. Sauvegarde la grille individuelle par mot  (_attention_grid.png)
          3. Sauvegarde l'overlay moyen                (_attention_overlay.png)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        out_dir = save_dir or 'output_attention'
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nImage : {image_path}")
        words, alphas = self.get_caption_and_attention(
            image_path, method=method, beam_width=beam_width, max_length=max_length
        )
        print(f'Caption : "{" ".join(words)}"')

        sw_count = sum(1 for w in words if is_stop_word(w))
        print(f"Stop words : {sw_count}/{len(words)} (affichés grisés dans la grille)")
        print(f"Grille     : {self.grid_size}×{self.grid_size} régions")

        basename     = os.path.splitext(os.path.basename(image_path))[0]
        path_grid    = os.path.join(out_dir, f'{basename}_attention_grid.png')
        path_overlay = os.path.join(out_dir, f'{basename}_attention_overlay.png')

        self.plot_attention_grid(
            image_path, words, alphas,
            save_path=path_grid, show_stop_words=show_stop_words
        )
        self.plot_attention_overlay(
            image_path, words, alphas,
            save_path=path_overlay
        )
        return words, alphas

    # ──────────────────────────────────────────────────────────────────────────

    def visualize_folder(self, image_dir, method='beam_search', beam_width=5,
                         max_length=20, max_images=None, save_dir=None):
        """Visualise l'attention pour toutes les images d'un dossier."""
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
        if max_images is not None:
            files = files[:max_images]

        print(f"\n{len(files)} image(s) trouvée(s) dans {image_dir}")

        results = {}
        for idx, fname in enumerate(files):
            path = os.path.join(image_dir, fname)
            print(f"\n[{idx+1}/{len(files)}] {fname}")
            try:
                result = self.visualize(
                    path, method=method, beam_width=beam_width,
                    max_length=max_length, save_dir=save_dir
                )
                if result:
                    results[fname] = ' '.join(result[0])
            except Exception as e:
                print(f"  Erreur : {e}")
                results[fname] = f"ERROR: {e}"

        return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise les cartes d\'attention sur les images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Source d'images (priorité décroissante) :
  --image       image unique (chemin explicite)
  --image_dir   dossier explicite
  (auto)        data/coco/test2017/ si présent
  (auto)        ImagesTest/ sinon

Exemples :
  python visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth
  python visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth --use_coco_test
  python visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image_dir ImagesTest/
  python visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image ImagesTest/dog.jpg
  python visualize_attention.py --checkpoint checkpoints/resnet/cosine/best_model.pth   --method greedy
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le fichier .pth du modèle (doit avoir de l\'attention : resnet ou densenet)')
    parser.add_argument('--vocab_path', type=str, default='data/coco_vocab.pkl')
    parser.add_argument('--image',      type=str, default=None,
                        help='Chemin vers une seule image (priorité max)')
    parser.add_argument('--image_dir',  type=str, default=None,
                        help='Dossier contenant des images (priorité 2)')
    parser.add_argument('--use_coco_test', action='store_true',
                        help=f'Forcer l\'utilisation de {COCO_TEST_DIR}')
    parser.add_argument('--method',     choices=['greedy', 'beam_search'],
                        default='beam_search')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--save_dir',   type=str, default='output_attention',
                        help='Dossier de sortie (défaut: output_attention/)')
    parser.add_argument('--show_stop_words', action='store_true',
                        help='Inclure les mots de liaison dans la grille')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Résolution de la source d'images ─────────────────────────────────────
    if args.use_coco_test:
        if not os.path.isdir(COCO_TEST_DIR):
            print(f"Erreur : {COCO_TEST_DIR} introuvable.")
            print("Téléchargez d'abord : http://images.cocodataset.org/zips/test2017.zip")
            return
        mode, source = 'folder', COCO_TEST_DIR
        print(f"[Forcé] Source : {COCO_TEST_DIR}")
    else:
        mode, source = resolve_image_source(args)

    if mode is None:
        print("Erreur : aucune source d'images trouvée.")
        print(f"  Créez {FALLBACK_TEST_DIR}/ ou téléchargez {COCO_TEST_DIR}/")
        print("  Ou spécifiez --image <chemin> / --image_dir <dossier>")
        return

    # ── Résolution du checkpoint ──────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        print(f"Erreur : checkpoint introuvable → {ckpt_path}")
        return

    print(f"Checkpoint : {ckpt_path}")

    visualizer = AttentionVisualizer(
        checkpoint_path=ckpt_path,
        vocab_path=args.vocab_path,
    )

    if mode == 'single':
        visualizer.visualize(
            source,
            method=args.method, beam_width=args.beam_width,
            max_length=args.max_length, save_dir=args.save_dir,
            show_stop_words=args.show_stop_words
        )
    else:
        results = visualizer.visualize_folder(
            source,
            method=args.method, beam_width=args.beam_width,
            max_length=args.max_length, max_images=args.max_images,
            save_dir=args.save_dir
        )
        print(f"\n{'='*70}")
        print(f"RÉSUMÉ : {len(results)} image(s) traitée(s)  [source : {source}]")
        for img, cap in results.items():
            print(f"  {img} → {cap}")


if __name__ == "__main__":
    main()