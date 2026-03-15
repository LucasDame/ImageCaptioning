"""
demo.py — Génération de captions pour des images
=================================================

Utilisation :
    # Une seule image
    python demo.py --model densenet --image ImagesTest/dog.jpg

    # Toutes les images d'un dossier
    python demo.py --model densenet --image_dir ImagesTest/

    # Modèle spécifique (par défaut : best_model_cider.pth puis best_model.pth)
    python demo.py --model resnet --checkpoint checkpoints/resnet/cosine/best_model.pth

    # Méthode de génération
    python demo.py --model densenet --image_dir ImagesTest/ --method greedy
    python demo.py --model densenet --image_dir ImagesTest/ --method beam_search --beam_width 5
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

import torch
from PIL import Image
from torchvision import transforms

from config import get_config
from models.caption_model import load_model
from utils.vocabulary import Vocabulary
from utils.preprocessing import ImagePreprocessor


# =============================================================================
# CLASSE DEMO
# =============================================================================

class CaptionDemo:
    """
    Génère des captions pour des images en utilisant un modèle entraîné.
    Compatible avec les trois architectures (cnn / resnet / densenet).
    Le type est détecté automatiquement depuis le checkpoint.
    """

    def __init__(self, checkpoint_path, vocab_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device : {self.device}")

        print(f"\nChargement du modèle depuis {checkpoint_path}...")
        self.model, info = load_model(checkpoint_path, device=self.device)

        # Vocabulaire : depuis le checkpoint ou depuis un fichier séparé
        self.vocabulary = info.get('vocab')
        if self.vocabulary is None:
            if vocab_path is None:
                raise ValueError(
                    "Vocabulaire absent du checkpoint. "
                    "Spécifiez --vocab_path data/coco_vocab.pkl"
                )
            print(f"Chargement du vocabulaire depuis {vocab_path}...")
            self.vocabulary = Vocabulary.load(vocab_path)

        # Préprocesseur val (sans augmentation)
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

        print("✓ Modèle prêt !")

    # ──────────────────────────────────────────────────────────────────────────

    def generate_caption(self, image_path, method='beam_search', beam_width=5,
                         max_length=20):
        """
        Génère une caption pour une image.

        Args:
            image_path  : chemin vers le fichier image
            method      : 'greedy' ou 'beam_search'
            beam_width  : largeur du beam (beam_search uniquement)
            max_length  : longueur maximale de la caption

        Returns:
            str : caption générée
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image = self.image_prep(image_path, is_training=False)
        image = image.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if method == 'beam_search':
                features = self.model.encoder(image)
                caption_indices = self.model.decoder.generate_beam_search(
                    features, beam_width=beam_width,
                    max_length=max_length,
                    start_token=self.start_token, end_token=self.end_token
                )
            else:
                caption_indices = self.model.generate_caption(
                    image, max_length=max_length,
                    start_token=self.start_token, end_token=self.end_token,
                    method='greedy'
                )

        tokens = [
            t.item() if torch.is_tensor(t) else t
            for t in caption_indices[0]
            if (t.item() if torch.is_tensor(t) else t)
            not in [self.start_token, self.end_token, self.pad_token]
        ]
        return self.vocabulary.denumericalize(tokens)

    # ──────────────────────────────────────────────────────────────────────────

    def _show_or_save(self, fig, save_dir=None, filename=None):
        """Affiche ou sauvegarde la figure."""
        if _DISPLAY_MODE == 'interactive' and save_dir is None:
            plt.show()
        else:
            out_dir = save_dir or 'output_captions'
            os.makedirs(out_dir, exist_ok=True)
            fname = filename or 'caption.png'
            path  = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  → Sauvegardé : {path}")
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────

    def demo_single(self, image_path, method='beam_search', beam_width=5,
                    max_length=20, save_dir=None):
        """Génère et affiche la caption pour une seule image."""
        print(f"\nImage : {image_path}")
        caption = self.generate_caption(image_path, method=method,
                                        beam_width=beam_width,
                                        max_length=max_length)
        print(f'Caption : "{caption}"')

        img = Image.open(image_path).convert('RGB')
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'"{caption}"', fontsize=16, fontweight='bold', pad=20, wrap=True)
        plt.tight_layout()

        fname = os.path.splitext(os.path.basename(image_path))[0] + '_caption.png'
        self._show_or_save(fig, save_dir=save_dir, filename=fname)
        return caption

    # ──────────────────────────────────────────────────────────────────────────

    def demo_folder(self, image_dir, method='beam_search', beam_width=5,
                    max_length=20, max_images=None, save_dir=None):
        """Génère des captions pour toutes les images d'un dossier."""
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
                caption = self.generate_caption(path, method=method,
                                                beam_width=beam_width,
                                                max_length=max_length)
                print(f'  Caption : "{caption}"')
                results[fname] = caption

                img = Image.open(path).convert('RGB')
                fig = plt.figure(figsize=(8, 7))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    f'[{idx+1}/{len(files)}] {fname}\n\n"{caption}"',
                    fontsize=13, fontweight='bold', pad=15
                )
                plt.tight_layout()
                out_fname = os.path.splitext(fname)[0] + '_caption.png'
                self._show_or_save(fig, save_dir=save_dir, filename=out_fname)

            except Exception as e:
                print(f"  Erreur : {e}")
                results[fname] = f"ERROR: {e}"

        return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Génère des captions pour des images avec un modèle entraîné.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python demo.py --model densenet --image ImagesTest/dog.jpg
  python demo.py --model resnet   --image_dir ImagesTest/
  python demo.py --model cnn      --image_dir ImagesTest/ --method greedy
  python demo.py --model densenet --checkpoint checkpoints/densenet/cosine/best_model_cider.pth --image_dir ImagesTest/
        """
    )
    parser.add_argument('--model',      choices=['cnn', 'resnet', 'densenet'],
                        default='densenet',
                        help='Architecture (défaut: densenet) — détermine le dossier checkpoint')
    parser.add_argument('--scheduler',  choices=['plateau', 'cosine'],
                        default='cosine',
                        help='Scheduler utilisé à l\'entraînement (défaut: cosine) — '
                             'détermine le sous-dossier checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Chemin explicite vers le checkpoint (remplace --model/--scheduler)')
    parser.add_argument('--vocab_path', type=str, default='data/coco_vocab.pkl',
                        help='Chemin vers le vocabulaire si absent du checkpoint')
    parser.add_argument('--image',      type=str, default=None,
                        help='Chemin vers une seule image')
    parser.add_argument('--image_dir',  type=str, default=None,
                        help='Dossier contenant plusieurs images')
    parser.add_argument('--method',     choices=['greedy', 'beam_search'],
                        default='beam_search',
                        help='Méthode de génération (défaut: beam_search)')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Largeur du beam (défaut: 5)')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Longueur maximale de la caption (défaut: 20)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Nombre max d\'images à traiter (image_dir uniquement)')
    parser.add_argument('--save_dir',   type=str, default='output_captions',
                        help='Dossier de sortie des figures (défaut: output_captions/)')
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
        # Priorité : best_model_cider.pth > best_model.pth
        for fname in ['best_model_cider.pth', 'best_model.pth']:
            candidate = os.path.join(base, fname)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        else:
            print(f"Erreur : aucun checkpoint trouvé dans {base}")
            print("Lancez d'abord : python train.py --model "
                  f"{args.model} --scheduler {args.scheduler}")
            return

    print(f"Checkpoint : {ckpt_path}")

    demo = CaptionDemo(
        checkpoint_path=ckpt_path,
        vocab_path=args.vocab_path,
    )

    if args.image:
        demo.demo_single(
            args.image,
            method=args.method, beam_width=args.beam_width,
            max_length=args.max_length, save_dir=args.save_dir
        )
    else:
        results = demo.demo_folder(
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