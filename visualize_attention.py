"""
visualize_attention.py — Visualisation de l'attention
======================================================

Disponible uniquement avec model='resnet' ou model='densenet'.
Le type est détecté automatiquement depuis le checkpoint.

Utilisation :
    python visualize_attention.py --model densenet --image ImagesTest/dog.jpg
    python visualize_attention.py --model resnet   --image_dir ImagesTest/
    python visualize_attention.py --model densenet --image ImagesTest/dog.jpg --method greedy
    python visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image ImagesTest/dog.jpg
"""

import argparse
import math
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

    def visualize(self, image_path, method='beam_search', beam_width=5,
                  max_length=20, save_dir=None, show_stop_words=False):
        """
        Crée la visualisation complète :
          - Grille mot-par-mot avec cartes d'attention
          - Overlay de l'attention moyenne (mots de contenu uniquement)

        Args:
            image_path      : chemin vers l'image
            method          : 'greedy' ou 'beam_search'
            beam_width      : largeur du beam
            max_length      : longueur max de la caption
            save_dir        : dossier de sauvegarde (None = output_attention/)
            show_stop_words : inclure les mots de liaison dans la grille
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        # Créer le dossier de sortie dès maintenant (même si on retourne tôt)
        out_dir = save_dir or 'output_attention'
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nImage : {image_path}")
        words, alphas = self.get_caption_and_attention(
            image_path, method=method, beam_width=beam_width, max_length=max_length
        )
        print(f'Caption : "{" ".join(words)}"')

        img_orig = Image.open(image_path).convert('RGB')
        img_arr  = np.array(img_orig.resize((224, 224)))

        # Filtrer les mots à afficher
        display_items = []
        for i, word in enumerate(words):
            if i >= alphas.shape[0]:
                break
            alpha = alphas[i].cpu().numpy().reshape(self.grid_size, self.grid_size)
            is_stop = is_stop_word(word)
            if not is_stop or show_stop_words:
                display_items.append((word, alpha, is_stop))

        # Si aucun mot de contenu, retomber sur tous les mots (stop words inclus)
        if not display_items and words:
            for i, word in enumerate(words):
                if i >= alphas.shape[0]:
                    break
                alpha = alphas[i].cpu().numpy().reshape(self.grid_size, self.grid_size)
                display_items.append((word, alpha, True))

        if not display_items:
            print("Aucun mot à visualiser.")
            return

        n_words  = len(display_items)
        n_cols   = min(6, n_words)
        n_rows   = math.ceil(n_words / n_cols) + 1  # +1 pour l'overlay

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        if n_rows == 1:
            axes = [axes]
        axes = [row if hasattr(row, '__len__') else [row] for row in axes]
        # Aplatir
        all_axes = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

        # Désactiver tous les axes d'abord
        for ax in all_axes:
            ax.axis('off')

        # Grille mot-par-mot
        for idx, (word, alpha, is_stop) in enumerate(display_items):
            ax = all_axes[idx]
            # Upscale de la carte d'attention
            alpha_up = np.array(Image.fromarray(
                (alpha * 255).astype(np.uint8)
            ).resize((224, 224), resample=Image.BILINEAR)) / 255.0

            ax.imshow(img_arr)
            ax.imshow(alpha_up, alpha=0.5 if not is_stop else 0.25,
                      cmap='jet', vmin=0, vmax=1)
            ax.set_title(
                word,
                fontsize=11,
                fontweight='bold' if not is_stop else 'normal',
                color='black' if not is_stop else '#888888',
                pad=4
            )
            if is_stop:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc')
                ax.set_facecolor('#f5f5f5')
            ax.axis('off')

        # Overlay moyen (mots de contenu uniquement)
        content_alphas = [alpha for word, alpha, is_stop in display_items
                          if not is_stop]
        if content_alphas:
            mean_alpha = np.mean(content_alphas, axis=0)
            mean_alpha_up = np.array(Image.fromarray(
                (mean_alpha * 255).astype(np.uint8)
            ).resize((224, 224), resample=Image.BILINEAR)) / 255.0

            # Placer l'overlay dans la dernière rangée, centré
            last_row_start = n_cols * (n_rows - 1)
            overlay_ax_idx = last_row_start + (n_cols // 2)
            if overlay_ax_idx < len(all_axes):
                ax_ov = all_axes[overlay_ax_idx]
                ax_ov.imshow(img_arr)
                ax_ov.imshow(mean_alpha_up, alpha=0.55, cmap='jet', vmin=0, vmax=1)
                ax_ov.set_title('Attention moyenne\n(mots de contenu)',
                                fontsize=11, fontweight='bold', color='darkred', pad=4)
                ax_ov.axis('on')
                ax_ov.set_xticks([]); ax_ov.set_yticks([])
                for spine in ax_ov.spines.values():
                    spine.set_edgecolor('darkred')
                    spine.set_linewidth(2)

        caption_str = ' '.join(words)
        fig.suptitle(f'"{caption_str}"', fontsize=13, fontweight='bold', y=1.01, wrap=True)
        plt.tight_layout()

        # Sauvegarde (out_dir déjà défini et créé en début de méthode)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(out_dir, f'{basename}_attention.png')

        if _DISPLAY_MODE == 'interactive' and save_dir is None:
            plt.show()
        else:
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {out_path}")

        plt.close(fig)
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
Exemples :
  python visualize_attention.py --model densenet --image ImagesTest/dog.jpg
  python visualize_attention.py --model resnet   --image_dir ImagesTest/
  python visualize_attention.py --model densenet --image_dir ImagesTest/ --method greedy
        """
    )
    parser.add_argument('--model',      choices=['resnet', 'densenet'],
                        default='densenet',
                        help='Architecture (doit avoir de l\'attention)')
    parser.add_argument('--scheduler',  choices=['plateau', 'cosine'],
                        default='cosine',
                        help='Scheduler utilisé à l\'entraînement (défaut: cosine)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint explicite (remplace --model/--scheduler)')
    parser.add_argument('--vocab_path', type=str, default='data/coco_vocab.pkl')
    parser.add_argument('--image',      type=str, default=None,
                        help='Chemin vers une seule image')
    parser.add_argument('--image_dir',  type=str, default=None,
                        help='Dossier contenant des images')
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

    if args.image is None and args.image_dir is None:
        print("Erreur : spécifiez --image ou --image_dir.")
        return

    # Résolution du checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        config = get_config(args.model)
        base   = os.path.join(config['checkpoint_dir'], args.scheduler)
        for fname in ['best_model_cider.pth', 'best_model.pth']:
            candidate = os.path.join(base, fname)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        else:
            print(f"Erreur : aucun checkpoint trouvé dans {base}")
            print(f"Lancez d'abord : python train.py --model {args.model} --scheduler {args.scheduler}")
            return

    print(f"Checkpoint : {ckpt_path}")

    visualizer = AttentionVisualizer(
        checkpoint_path=ckpt_path,
        vocab_path=args.vocab_path,
    )

    if args.image:
        visualizer.visualize(
            args.image,
            method=args.method, beam_width=args.beam_width,
            max_length=args.max_length, save_dir=args.save_dir,
            show_stop_words=args.show_stop_words
        )
    else:
        results = visualizer.visualize_folder(
            args.image_dir,
            method=args.method, beam_width=args.beam_width,
            max_length=args.max_length, max_images=args.max_images,
            save_dir=args.save_dir
        )
        print(f"\n{'='*70}")
        print(f"RÉSUMÉ : {len(results)} image(s) traitée(s)")
        for img, cap in results.items():
            print(f"  {img} → {cap}")


if __name__ == "__main__":
    main()