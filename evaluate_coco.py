"""
evaluate.py — Scores finaux ResNet vs DenseNet sur COCO val2017
================================================================

Calcule BLEU-1, BLEU-4, METEOR et CIDEr (5 refs/image) sur l'intégralité
du split val2017 (~5 000 images) pour les deux modèles entraînés.

Utilisation CLI :
    # Les deux modèles (défaut)
    python evaluate.py

    # Un seul modèle
    python evaluate.py --model resnet
    python evaluate.py --model densenet

    # Checkpoints personnalisés
    python evaluate.py \\
        --resnet_ckpt  checkpoints_coco2/best_model_cider.pth \\
        --densenet_ckpt checkpoints_coco3/best_model_cider.pth

    # Évaluation rapide sur 500 images
    python evaluate.py --num_samples 500

    # Exporter les captions générées
    python evaluate.py --save_captions results/captions.json

Utilisation depuis un notebook / script :
    from evaluate import evaluate_models
    results = evaluate_models()
    results = evaluate_models(num_samples=500, generation_method='greedy')
"""

import os
import json
import math
import time
import argparse
from collections import Counter

import torch
from torchvision import transforms
from tqdm import tqdm

# ── Métriques NLTK ────────────────────────────────────────────────────────────
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

# ── Imports projet ─────────────────────────────────────────────────────────────
from utils.vocabulary import Vocabulary
from utils.preprocessing_coco import CaptionPreprocessor, ImagePreprocessor
from utils.data_loader import get_data_loaders
from models2.caption_model2 import load_model


# =============================================================================
# CIDEr — identique à train_coco2/3
# =============================================================================

def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def compute_cider_score(generated_list, reference_list, n_max=4):
    """
    CIDEr-D sur corpus complet.
      generated_list : list[list[str]]
      reference_list : list[list[list[str]]]  — jusqu'à 5 refs/image
    """
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
        all_ng  = set(tf_gen) | set(tf_ref)
        vec_gen = {ng: tf_gen.get(ng, 0) * idf.get((n, ng), 0.) for ng in all_ng}
        vec_ref = {ng: tf_ref.get(ng, 0) * idf.get((n, ng), 0.) for ng in all_ng}
        return vec_gen, vec_ref

    scores = []
    for gen_words, refs in zip(generated_list, reference_list):
        score_n = []
        for n in range(1, n_max + 1):
            vg, vr   = tfidf_vec(gen_words, refs, n)
            dot      = sum(vg.get(ng, 0) * vr.get(ng, 0) for ng in vr)
            norm_gen = math.sqrt(sum(v ** 2 for v in vg.values())) + 1e-10
            norm_ref = math.sqrt(sum(v ** 2 for v in vr.values())) + 1e-10
            bp = (math.exp(1 - len(refs[0]) / (len(gen_words) + 1e-10))
                  if gen_words and refs and len(gen_words) < len(refs[0]) else 1.0)
            score_n.append(bp * dot / (norm_gen * norm_ref))
        scores.append(sum(score_n) / n_max)

    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# TRANSFORM DE TEST (identique à la validation dans train_coco2/3)
# =============================================================================

_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# GÉNÉRATION + MÉTRIQUES
# =============================================================================

def _build_val_refs(val_pairs):
    """
    Construit le dict image_path → [ref_words_1, ..., ref_words_5]
    depuis la liste plate val_pairs.
    """
    val_refs = {}
    for pair in val_pairs:
        path  = pair['image_path']
        words = [w for w in pair['caption'].lower().split()
                 if w not in {'', '.', ',', '!', '?'}]
        if words:
            if path not in val_refs:
                val_refs[path] = []
            val_refs[path].append(words)
    return val_refs


def _collect_and_score(model, val_loader, val_refs, vocab,
                       generation_method, beam_width,
                       num_samples, device):
    """
    Une seule passe sur val_loader :
      - génère une caption par image (seen_paths évite les doublons)
      - récupère les 5 refs depuis val_refs
    Retourne generated_list, reference_list, caption_log.
    """
    start_token = vocab.word2idx[vocab.start_token]
    end_token   = vocab.word2idx[vocab.end_token]
    pad_token   = vocab.word2idx[vocab.pad_token]
    specials    = {start_token, end_token, pad_token}

    def decode(tok_list):
        words = []
        for t in tok_list:
            if t == end_token:
                break
            if t not in specials:
                words.append(vocab.idx2word.get(t, '<unk>'))
        return words

    dataset        = val_loader.dataset
    generated_list = []
    reference_list = []
    caption_log    = []
    seen_paths     = set()
    global_idx     = 0
    limit          = num_samples or float('inf')

    model.eval()
    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader,
                                              desc='  Inférence', unit='batch'):
            if len(generated_list) >= limit:
                break
            images_gpu = images.to(device)

            for i in range(images_gpu.size(0)):
                if len(generated_list) >= limit:
                    break

                # shuffle=False → dataset.pairs[global_idx] correspond à cet item
                try:
                    image_path = dataset.pairs[global_idx]['image_path']
                except (AttributeError, IndexError, KeyError):
                    image_path = None
                global_idx += 1

                # Une seule génération par image (val expose 5× chaque image)
                if image_path in seen_paths:
                    continue

                # Références : 5/image depuis val_refs, 1 en fallback
                if image_path and image_path in val_refs:
                    refs = val_refs[image_path]
                else:
                    ref_ids   = [t.item() for t in captions[i]
                                 if t.item() not in specials]
                    ref_words = decode(ref_ids)
                    refs      = [ref_words] if ref_words else None

                if not refs:
                    continue

                # ── Génération ──────────────────────────────────────────────
                features = model.encoder(images_gpu[i:i+1])

                if generation_method == 'beam_search':
                    tokens = model.decoder.generate_beam_search(
                        features,
                        beam_width  = beam_width,
                        max_length  = 20,
                        start_token = start_token,
                        end_token   = end_token,
                    )
                else:
                    tokens = model.decoder.generate(
                        features,
                        max_length  = 20,
                        start_token = start_token,
                        end_token   = end_token,
                    )

                tok_list = (tokens[0].tolist() if tokens.dim() == 2
                            else tokens.tolist())
                gen_words = decode(tok_list)

                if not gen_words:
                    continue

                generated_list.append(gen_words)
                reference_list.append(refs)
                if image_path:
                    seen_paths.add(image_path)
                    caption_log.append({
                        'image_path': image_path,
                        'generated':  ' '.join(gen_words),
                        'references': [' '.join(r) for r in refs],
                    })

    return generated_list, reference_list, caption_log


def _compute_metrics(generated_list, reference_list):
    scores = {}

    if BLEU_AVAILABLE and generated_list:
        smooth           = SmoothingFunction().method1
        scores['BLEU-1'] = corpus_bleu(reference_list, generated_list,
                                       weights=(1, 0, 0, 0),
                                       smoothing_function=smooth)
        scores['BLEU-4'] = corpus_bleu(reference_list, generated_list,
                                       weights=(.25, .25, .25, .25),
                                       smoothing_function=smooth)
    else:
        scores['BLEU-1'] = scores['BLEU-4'] = None

    if METEOR_AVAILABLE and generated_list:
        vals             = [meteor_score(refs, gen)
                            for gen, refs in zip(generated_list, reference_list)]
        scores['METEOR'] = sum(vals) / len(vals)
    else:
        scores['METEOR'] = None

    scores['CIDEr'] = compute_cider_score(generated_list, reference_list)
    return scores


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def evaluate_models(
    # Checkpoints
    resnet_ckpt      = 'checkpoints_coco2/best_model.pth',
    densenet_ckpt    = 'checkpoints_coco3/best_model.pth',
    # Données
    val_captions_file = 'data/coco/annotations/captions_val2017.json',
    val_images_dir    = 'data/coco/val2017',
    vocab_path        = 'data/coco_vocab.pkl',
    # Évaluation
    num_samples       = None,
    batch_size        = 32,
    num_workers       = 4,
    generation_method = 'beam_search',
    beam_width        = 5,
    # Sorties
    save_captions     = None,
    device            = None,
):
    """
    Évalue les deux modèles (ResNet attention et DenseNet-121) sur val2017.

    Passer None à resnet_ckpt ou densenet_ckpt pour ignorer ce modèle.

    Retourne :
        dict { nom_modèle: { 'BLEU-1', 'BLEU-4', 'METEOR', 'CIDEr',
                             'n_images', 'time_s' } }

    Exemple :
        from evaluate import evaluate_models
        results = evaluate_models()
        results = evaluate_models(num_samples=500, generation_method='greedy')
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print("=" * 70)
    print("ÉVALUATION FINALE — ResNet (attention) vs DenseNet-121")
    print(f"Device : {device}")
    print("=" * 70)

    # ── 1. Données ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Chargement des données val2017 ...")
    prep      = CaptionPreprocessor(captions_file=val_captions_file,
                                    images_dir=val_images_dir)
    val_pairs = prep.get_image_caption_pairs()
    val_refs  = _build_val_refs(val_pairs)

    n_images = len(val_refs)
    avg_refs = sum(len(v) for v in val_refs.values()) / max(n_images, 1)
    print(f"      {len(val_pairs)} paires  |  {n_images} images uniques"
          f"  |  {avg_refs:.1f} refs/image")

    # ── 2. DataLoader (shuffle=False garanti par get_data_loaders) ────────────
    print(f"\n[2/3] DataLoader ...")
    vocab      = Vocabulary.load(vocab_path)
    image_prep = ImagePreprocessor(
        image_size=224, normalize=False,
        train_transform=_TEST_TRANSFORM,
        val_transform=_TEST_TRANSFORM,
    )
    # val_pairs dans les deux slots : seul val_loader (shuffle=False) est utilisé
    _, val_loader = get_data_loaders(
        train_pairs   = val_pairs[:1],      # dummy train — non utilisé
        val_pairs     = val_pairs,
        vocabulary    = vocab,
        image_preprocessor = image_prep,
        batch_size    = batch_size,
        num_workers   = num_workers,
        shuffle_train = False,
    )

    # ── 3. Évaluation ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Génération ({generation_method}"
          + (f", beam={beam_width}" if generation_method == 'beam_search' else "")
          + f") sur {num_samples or n_images} images ...")

    models_to_eval = []
    if resnet_ckpt:
        models_to_eval.append(('ResNet (attention)', resnet_ckpt))
    if densenet_ckpt:
        models_to_eval.append(('DenseNet-121', densenet_ckpt))

    if not models_to_eval:
        print("⚠️  Aucun checkpoint fourni.")
        return {}

    all_results = {}
    all_logs    = {}

    for model_name, ckpt_path in models_to_eval:
        print(f"\n{'─' * 70}")
        print(f"  Modèle     : {model_name}")
        print(f"  Checkpoint : {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"  ✗ Fichier introuvable")
            all_results[model_name] = None
            continue

        t0 = time.time()
        try:
            # load_model reconstruit automatiquement le bon encoder (densenet
            # ou attention) et restaure growth_rate/block_config depuis le ckpt
            model, info = load_model(ckpt_path, device=str(device))
            model.eval()

            # Vocab embarqué dans le checkpoint (sinon fallback sur vocab_path)
            used_vocab = info.get('vocab') or vocab

            generated_list, reference_list, caption_log = _collect_and_score(
                model=model,
                val_loader=val_loader,
                val_refs=val_refs,
                vocab=used_vocab,
                generation_method=generation_method,
                beam_width=beam_width,
                num_samples=num_samples,
                device=device,
            )

            scores             = _compute_metrics(generated_list, reference_list)
            scores['n_images'] = len(generated_list)
            scores['time_s']   = round(time.time() - t0, 1)
            all_results[model_name] = scores
            all_logs[model_name]    = caption_log

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(f"  ✗ Erreur : {exc}")
            import traceback; traceback.print_exc()
            all_results[model_name] = None

    # ── Tableau ───────────────────────────────────────────────────────────────
    _print_table(all_results, generation_method, beam_width)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    scores_path = 'results/evaluation_scores.json'
    with open(scores_path, 'w') as f:
        json.dump({
            'config': {
                'val_file':          val_captions_file,
                'num_samples':       num_samples,
                'generation_method': generation_method,
                'beam_width':        beam_width,
            },
            'results': {k: v for k, v in all_results.items() if v is not None},
        }, f, indent=4)
    print(f"\nScores sauvegardés → {scores_path}")

    if save_captions and all_logs:
        os.makedirs(os.path.dirname(os.path.abspath(save_captions)), exist_ok=True)
        with open(save_captions, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=2, ensure_ascii=False)
        print(f"Captions sauvegardées → {save_captions}")

    return all_results


# =============================================================================
# AFFICHAGE
# =============================================================================

_METRICS = ['BLEU-1', 'BLEU-4', 'METEOR', 'CIDEr']


def _print_table(all_results, generation_method, beam_width):
    print(f"\n{'=' * 70}")
    print("RÉSULTATS FINAUX  (val2017 COCO)")
    print(f"{'=' * 70}")

    W_MODEL  = 22
    W_METRIC = 11

    header = f"{'Modèle':<{W_MODEL}}"
    for m in _METRICS:
        header += f"  {m:>{W_METRIC}}"
    header += f"  {'Images':>8}  {'Temps':>7}"
    print(header)
    print("─" * len(header))

    valid = {k: v for k, v in all_results.items() if v is not None}
    best  = {m: max((v[m] for v in valid.values() if v.get(m) is not None),
                    default=-1)
             for m in _METRICS}

    for name, scores in all_results.items():
        if scores is None:
            row  = f"{name:<{W_MODEL}}"
            row += "".join(f"  {'N/A':>{W_METRIC}}" for _ in _METRICS)
            row += f"  {'—':>8}  {'—':>7}"
            print(row)
            continue

        row = f"{name:<{W_MODEL}}"
        for m in _METRICS:
            val = scores.get(m)
            if val is None:
                cell = "N/A"
            else:
                cell = f"{val:.4f}"
                if len(valid) > 1 and val == best[m]:
                    cell += " ★"
            row += f"  {cell:>{W_METRIC}}"

        n = scores.get('n_images', '?')
        t = scores.get('time_s', '?')
        row += f"  {str(n):>8}  {str(t) + 's':>7}"
        print(row)

    print("─" * len(header))
    gen_str = generation_method + (f" beam={beam_width}"
                                   if generation_method == 'beam_search' else "")
    print(f"★ = meilleur score  |  {gen_str}")
    print(f"{'=' * 70}")

    # Bilan si deux modèles évalués
    if len(valid) == 2:
        names  = list(valid.keys())
        s0, s1 = valid[names[0]], valid[names[1]]
        wins   = [0, 0]
        for m in _METRICS:
            v0, v1 = s0.get(m) or 0, s1.get(m) or 0
            if v0 > v1:   wins[0] += 1
            elif v1 > v0: wins[1] += 1
        print(f"\nBilan : {names[0]} l'emporte sur {wins[0]}/{len(_METRICS)} métriques, "
              f"{names[1]} sur {wins[1]}/{len(_METRICS)}.")
        c0, c1 = s0.get('CIDEr') or 0, s1.get('CIDEr') or 0
        if c0 > 0 and c1 > 0:
            winner = names[0] if c0 > c1 else names[1]
            pct    = abs(c0 - c1) / min(c0, c1) * 100
            print(f"CIDEr (métrique principale) : {winner} est {pct:.1f}% supérieur.\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Scores finaux BLEU/METEOR/CIDEr sur val2017 COCO'
    )
    p.add_argument('--model',
                   choices=['resnet', 'densenet', 'both'], default='both',
                   help='Modèle(s) à évaluer (défaut : both)')
    p.add_argument('--resnet_ckpt',
                   default='checkpoints_coco2/best_model.pth',
                   help='Checkpoint ResNet (train_coco2)')
    p.add_argument('--densenet_ckpt',
                   default='checkpoints_coco3/best_model.pth',
                   help='Checkpoint DenseNet (train_coco3)')
    p.add_argument('--val_captions',
                   default='data/coco/annotations/captions_val2017.json')
    p.add_argument('--val_images',
                   default='data/coco/val2017')
    p.add_argument('--vocab_path',
                   default='data/coco_vocab.pkl')
    p.add_argument('--num_samples', type=int, default=None,
                   help="Nombre d'images évaluées (défaut : toutes ~5000)")
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--generation_method',
                   choices=['greedy', 'beam_search'], default='beam_search')
    p.add_argument('--beam_width',  type=int, default=5)
    p.add_argument('--save_captions', default=None,
                   help='Chemin JSON pour exporter les captions générées')
    p.add_argument('--device', default=None, help='cuda | cpu (défaut : auto)')
    args = p.parse_args()

    evaluate_models(
        resnet_ckpt   = args.resnet_ckpt   if args.model in ('resnet',   'both') else None,
        densenet_ckpt = args.densenet_ckpt if args.model in ('densenet', 'both') else None,
        val_captions_file = args.val_captions,
        val_images_dir    = args.val_images,
        vocab_path        = args.vocab_path,
        num_samples       = args.num_samples,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        generation_method = args.generation_method,
        beam_width        = args.beam_width,
        save_captions     = args.save_captions,
        device            = args.device,
    )


if __name__ == '__main__':
    main()