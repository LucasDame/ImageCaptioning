"""
evaluate.py — Évaluation BLEU / METEOR / CIDEr sur COCO val2017
================================================================

Utilisation :
    # Évaluer un seul modèle
    python evaluate.py --model densenet --scheduler cosine

    # Comparer plusieurs modèles
    python evaluate.py --model densenet resnet cnn --scheduler cosine

    # Checkpoint spécifique
    python evaluate.py --checkpoint checkpoints/densenet/cosine/best_model_cider.pth

    # Rapide (500 images)
    python evaluate.py --model densenet --num_samples 500

    # Exporter les captions générées
    python evaluate.py --model densenet --save_captions results/captions.json
"""

import argparse
import json
import math
import os
import time
from collections import Counter

import torch
from torchvision import transforms
from tqdm import tqdm

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    for _res in ['tokenizers/punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(_res)
        except LookupError:
            nltk.download(_res.split('/')[-1], quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK non installé — BLEU/METEOR désactivés.  pip install nltk")
    BLEU_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False

from config import get_config
from models.caption_model import load_model
from utils.vocabulary import Vocabulary
from utils.preprocessing import CaptionPreprocessor, ImagePreprocessor
from utils.data_loader import get_data_loaders


# =============================================================================
# CIDEr
# =============================================================================

def _ngrams(words, n):
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def compute_cider_score(generated_list, reference_list, n_max=4):
    num_docs = len(reference_list)
    idf = {}
    for n in range(1, n_max + 1):
        doc_freq = Counter()
        for refs in reference_list:
            seen = set()
            for ref in refs:
                for ng in _ngrams(ref, n):
                    seen.add(ng)
            for ng in seen:
                doc_freq[ng] += 1
        for ng, df in doc_freq.items():
            idf[(n, ng)] = math.log((num_docs + 1.0) / (df + 1.0))

    def tfidf_vec(words, refs, n):
        tf_gen = _ngrams(words, n)
        tf_ref = Counter()
        for ref in refs:
            for ng, cnt in _ngrams(ref, n).items():
                tf_ref[ng] += cnt
        if refs:
            for ng in tf_ref:
                tf_ref[ng] /= len(refs)
        vec_gen, vec_ref = {}, {}
        for ng in set(tf_gen) | set(tf_ref):
            w = idf.get((n, ng), 0.0)
            vec_gen[ng] = tf_gen.get(ng, 0) * w
            vec_ref[ng] = tf_ref.get(ng, 0) * w
        return vec_gen, vec_ref

    scores = []
    for gen_words, refs in zip(generated_list, reference_list):
        score_n = []
        for n in range(1, n_max + 1):
            vec_gen, vec_ref = tfidf_vec(gen_words, refs, n)
            dot  = sum(vec_gen.get(ng, 0) * vec_ref.get(ng, 0) for ng in vec_ref)
            norm_gen = math.sqrt(sum(v**2 for v in vec_gen.values())) + 1e-10
            norm_ref = math.sqrt(sum(v**2 for v in vec_ref.values())) + 1e-10
            bp = 1.0
            if refs and len(gen_words) < len(refs[0]):
                bp = math.exp(1 - len(refs[0]) / (len(gen_words) + 1e-10))
            score_n.append(bp * dot / (norm_gen * norm_ref))
        scores.append(sum(score_n) / n_max)
    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# ÉVALUATION D'UN MODÈLE
# =============================================================================

def evaluate_model(checkpoint_path, val_pairs, vocab,
                   num_samples=5000, generation_method='beam_search',
                   beam_width=5, max_length=20, device=None):
    """
    Évalue un modèle sur le split val COCO.

    Args:
        checkpoint_path  : chemin vers le checkpoint .pth
        val_pairs        : paires image-caption de validation
        vocab            : objet Vocabulary
        num_samples      : nombre d'images à évaluer (None = tout le val set)
        generation_method: 'greedy' ou 'beam_search'
        beam_width       : largeur du beam
        max_length       : longueur max de la caption générée
        device           : torch.device

    Returns:
        dict : scores BLEU-1, BLEU-4, METEOR, CIDEr
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nChargement du modèle depuis {checkpoint_path}...")
    model, info = load_model(checkpoint_path, device=device)
    model.eval()

    # Récupérer le vocabulaire depuis le checkpoint si disponible
    _vocab = info.get('vocab') or vocab
    start_token = _vocab.word2idx[_vocab.start_token]
    end_token   = _vocab.word2idx[_vocab.end_token]
    pad_token   = _vocab.word2idx[_vocab.pad_token]

    # Transform val
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_prep = ImagePreprocessor(
        image_size=224, normalize=False, val_transform=val_transform
    )

    # DataLoader val (batch_size=1 pour simplicité)
    from utils.data_loader import get_data_loaders
    _, val_loader = get_data_loaders(
        train_pairs=val_pairs[:1], val_pairs=val_pairs,
        vocabulary=_vocab, image_preprocessor=image_prep,
        batch_size=1, num_workers=2, shuffle_train=False
    )

    # Construire le dictionnaire image_path → list[list[str]] (5 refs/image)
    val_refs = {}
    for pair in val_pairs:
        path   = pair['image_path']
        words  = [w for w in pair['caption'].lower().split()
                  if w not in {'', '.', ',', '!', '?'}]
        if words:
            val_refs.setdefault(path, []).append(words)

    generated_list = []
    reference_list = []
    seen_paths     = set()
    global_sample  = 0
    dataset        = val_loader.dataset

    print(f"Génération ({generation_method})...")
    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader, desc='Évaluation'):
            if num_samples and len(generated_list) >= num_samples:
                break
            images = images.to(device)

            try:
                image_path = dataset.pairs[global_sample]['image_path']
            except (AttributeError, IndexError, KeyError):
                image_path = None
            global_sample += 1

            if image_path in seen_paths:
                continue

            refs_for_image = val_refs.get(image_path)
            if not refs_for_image:
                continue

            features = model.encoder(images)
            if generation_method == 'beam_search':
                generated = model.decoder.generate_beam_search(
                    features, beam_width=beam_width,
                    max_length=max_length,
                    start_token=start_token, end_token=end_token
                )
            else:
                generated = model.decoder.generate(
                    features, max_length=max_length,
                    start_token=start_token, end_token=end_token
                )

            gen_ids = [
                t.item() if torch.is_tensor(t) else t
                for t in generated[0]
                if (t.item() if torch.is_tensor(t) else t)
                not in [start_token, end_token, pad_token]
            ]
            gen_words = _vocab.denumericalize(gen_ids).split()

            if gen_words:
                generated_list.append(gen_words)
                reference_list.append(refs_for_image)
                if image_path:
                    seen_paths.add(image_path)

    print(f"  {len(generated_list)} captions générées")

    # ── Calcul des scores ────────────────────────────────────────────────────
    scores = {}

    if BLEU_AVAILABLE and generated_list:
        smooth = SmoothingFunction().method1
        scores['BLEU-1'] = corpus_bleu(reference_list, generated_list,
                                       weights=(1, 0, 0, 0),
                                       smoothing_function=smooth)
        scores['BLEU-4'] = corpus_bleu(reference_list, generated_list,
                                       weights=(.25, .25, .25, .25),
                                       smoothing_function=smooth)
    else:
        scores['BLEU-1'] = scores['BLEU-4'] = None

    if METEOR_AVAILABLE and generated_list:
        meteor_scores = [meteor_score(refs, gen)
                         for gen, refs in zip(generated_list, reference_list)]
        scores['METEOR'] = sum(meteor_scores) / len(meteor_scores)
    else:
        scores['METEOR'] = None

    scores['CIDEr'] = compute_cider_score(generated_list, reference_list) \
                      if generated_list else None

    return scores, generated_list, reference_list


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Évalue un ou plusieurs modèles sur COCO val2017.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes d'utilisation :
  --checkpoint   évalue un checkpoint précis (chemin .pth)
  --model        évalue le/les meilleur(s) checkpoint(s) d'une architecture
                 (requiert aussi --scheduler)

Exemples :
  python evaluate.py --checkpoint checkpoints/densenet/cosine/best_model_cider.pth
  python evaluate.py --checkpoint checkpoints/resnet/plateau/best_model.pth --num_samples 500
  python evaluate.py --model densenet resnet cnn --scheduler cosine
  python evaluate.py --model densenet --scheduler plateau --method greedy
        """
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Chemin vers un fichier .pth (évalue ce checkpoint uniquement)')
    parser.add_argument('--model',      choices=['cnn', 'resnet', 'densenet'],
                        nargs='+', default=None,
                        help='Modèle(s) à évaluer — requiert --scheduler')
    parser.add_argument('--scheduler',  choices=['plateau', 'cosine'],
                        default=None,
                        help='Scheduler utilisé à l\'entraînement — requis avec --model')
    parser.add_argument('--vocab_path', type=str, default='data/coco_vocab.pkl',
                        help='Vocabulaire (défaut: data/coco_vocab.pkl)')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Nombre d\'images à évaluer (défaut: 5000 = tout le val)')
    parser.add_argument('--method',     choices=['greedy', 'beam_search'],
                        default='beam_search',
                        help='Méthode de génération (défaut: beam_search)')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--save_captions', type=str, default=None,
                        help='Sauvegarder les captions générées dans ce fichier JSON')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validation des arguments ──────────────────────────────────────────────
    if not args.checkpoint and not args.model:
        print("Erreur : spécifiez --checkpoint <chemin.pth> ou --model <nom> --scheduler <scheduler>")
        return
    if args.model and not args.scheduler:
        print("Erreur : --model requiert --scheduler (plateau ou cosine)")
        return
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        print(f"Erreur : checkpoint introuvable → {args.checkpoint}")
        return

    # Charger les données val
    config = get_config('densenet')  # chemins COCO identiques pour tous les modèles

    print("Chargement du vocabulaire...")
    vocab = Vocabulary.load(args.vocab_path)

    print("Chargement des paires val COCO...")
    val_cap_prep = CaptionPreprocessor(
        config['val_captions_file'], config['val_images_dir']
    )
    val_pairs = val_cap_prep.get_image_caption_pairs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # Construire la liste de checkpoints à évaluer
    checkpoints = []

    if args.checkpoint:
        checkpoints.append((os.path.basename(args.checkpoint), args.checkpoint))
    else:
        for model_name in args.model:
            cfg  = get_config(model_name)
            base = os.path.join(cfg['checkpoint_dir'], args.scheduler)
            # Priorité : best_model_cider.pth > best_model.pth
            for fname in ['best_model_cider.pth', 'best_model.pth']:
                candidate = os.path.join(base, fname)
                if os.path.exists(candidate):
                    checkpoints.append((f'{model_name}/{args.scheduler}/{fname}', candidate))
                    break
            else:
                print(f"⚠️  Aucun checkpoint trouvé pour {model_name}/{args.scheduler}")

    if not checkpoints:
        print("Erreur : aucun checkpoint à évaluer.")
        return

    # ── Évaluation ────────────────────────────────────────────────────────────
    all_results   = {}
    all_captions  = {}
    start_time    = time.time()

    for label, ckpt_path in checkpoints:
        print(f"\n{'='*70}")
        print(f"Évaluation : {label}")
        print(f"{'='*70}")

        try:
            scores, gen_list, ref_list = evaluate_model(
                ckpt_path, val_pairs, vocab,
                num_samples=args.num_samples,
                generation_method=args.method,
                beam_width=args.beam_width,
                max_length=args.max_length,
                device=device
            )
            all_results[label] = scores
            if args.save_captions:
                all_captions[label] = gen_list

        except Exception as e:
            print(f"  Erreur : {e}")
            all_results[label] = {}

    # ── Tableau comparatif ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"RÉSULTATS ({args.num_samples} images, {args.method})")
    print(f"{'='*70}")

    col_w = 14
    header = f"{'Modèle':<30}" + ''.join(f"{'Métrique':>{col_w}}"
                                          for métrique in ['BLEU-1', 'BLEU-4', 'METEOR', 'CIDEr'])
    header = f"{'Modèle':<30}{'BLEU-1':>{col_w}}{'BLEU-4':>{col_w}}{'METEOR':>{col_w}}{'CIDEr':>{col_w}}"
    print(header)
    print('-' * (30 + 4 * col_w))

    for label, scores in all_results.items():
        row = f"{label:<30}"
        for m in ['BLEU-1', 'BLEU-4', 'METEOR', 'CIDEr']:
            v = scores.get(m)
            row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}}"
        print(row)

    print(f"\nTemps total : {elapsed/60:.2f} min")

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    os.makedirs(config['results_dir'], exist_ok=True)

    results_path = os.path.join(config['results_dir'], 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'num_samples':      args.num_samples,
                'generation_method': args.method,
                'beam_width':       args.beam_width,
            },
            'results': all_results,
        }, f, indent=4)
    print(f"\nRésultats sauvegardés → {results_path}")

    if args.save_captions and all_captions:
        with open(args.save_captions, 'w') as f:
            json.dump(all_captions, f, indent=2)
        print(f"Captions sauvegardées → {args.save_captions}")


if __name__ == "__main__":
    main()