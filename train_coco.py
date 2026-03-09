"""
Script d'entraînement COCO pour Image Captioning
=================================================

Compatible avec les trois encoder_type du nouveau modèle :
  'lite'      → EncoderCNNLite  + DecoderLSTM
  'full'      → EncoderCNN      + DecoderLSTM  (résiduel)
  'attention' → EncoderSpatial  + DecoderWithAttention

Différences avec train.py (Flickr8k) :
  - Deux fichiers JSON séparés (train2017 / val2017) au lieu d'un split manuel
  - Import depuis preprocessing_coco et config_coco
  - attention_dim passé à create_model si encoder_type='attention'
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import math
from collections import defaultdict
from torchvision import transforms

# ── Métriques BLEU (NLTK) ─────────────────────────────────────────────────────
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
    print("⚠️  NLTK non installé — BLEU désactivé. pip install nltk")
    BLEU_AVAILABLE = False

# ── Métrique METEOR (NLTK) ────────────────────────────────────────────────────
try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False

# ── Métrique CIDEr (implémentation légère sans dépendance externe) ─────────────
# CIDEr mesure la pertinence d'une caption générée par rapport aux références
# en pondérant les n-grammes par leur TF-IDF sur tout le corpus de validation.
# Plus le n-gramme est rare dans le corpus mais présent dans la référence, plus
# il compte. Valeur typique : 0.0 → 1.5+ (pas bornée à 1).
CIDER_AVAILABLE = True  # implémentation interne, toujours disponible


from utils import vocabulary, data_loader
from utils.preprocessing_coco import CaptionPreprocessor, ImagePreprocessor
from models2 import caption_model2

from config_coco import CONFIG


# =============================================================================
# IMPLÉMENTATION CIDER LÉGÈRE
# =============================================================================

def _ngrams(words, n):
    """Retourne un dict Counter de n-grammes pour une liste de mots."""
    from collections import Counter
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def compute_cider_score(generated_list, reference_list, n_max=4):
    """
    Calcule le score CIDEr-D sur un corpus.

    CIDEr (Consensus-based Image Description Evaluation) évalue une caption
    générée en la comparant à plusieurs références humaines. Il utilise
    TF-IDF pour donner plus de poids aux n-grammes informatifs (rares dans
    le corpus mais présents dans les références).

    Args:
        generated_list (list[list[str]]): Captions générées, chaque caption
                                          est une liste de mots.
        reference_list (list[list[list[str]]]): Références, chaque entrée est
                                                une liste de captions de référence
                                                (elles-mêmes listes de mots).
        n_max (int): Ordre maximum des n-grammes (défaut : 4, comme l'article).

    Returns:
        float: Score CIDEr moyen sur le corpus (≥ 0, typiquement 0–1.5).
    """
    from collections import Counter
    import math as _math

    num_refs = len(reference_list)

    # ── Étape 1 : calculer le IDF de chaque n-gramme sur tout le corpus ──
    # IDF(ng) = log( N / df(ng) )  où df = nombre de documents contenant ng
    idf = {}
    for n in range(1, n_max + 1):
        doc_freq = Counter()
        for refs in reference_list:
            # union des n-grammes de toutes les références d'une image
            seen = set()
            for ref in refs:
                for ng in _ngrams(ref, n):
                    seen.add(ng)
            for ng in seen:
                doc_freq[ng] += 1
        for ng, df in doc_freq.items():
            idf[(n, ng)] = _math.log((num_refs + 1.0) / (df + 1.0))

    def tfidf_vec(words, refs_for_image, n):
        """
        Vecteur TF-IDF pour les n-grammes d'une caption (words) par rapport
        aux références de la même image.
        """
        tf_gen = _ngrams(words, n)
        # TF moyen des références
        tf_ref = Counter()
        for ref in refs_for_image:
            for ng, cnt in _ngrams(ref, n).items():
                tf_ref[ng] += cnt
        if refs_for_image:
            for ng in tf_ref:
                tf_ref[ng] /= len(refs_for_image)

        vec_gen = {}
        vec_ref = {}
        all_ng = set(tf_gen) | set(tf_ref)
        for ng in all_ng:
            w = idf.get((n, ng), 0.0)
            vec_gen[ng] = tf_gen.get(ng, 0) * w
            vec_ref[ng] = tf_ref.get(ng, 0) * w
        return vec_gen, vec_ref

    scores = []
    for gen_words, refs in zip(generated_list, reference_list):
        score_n = []
        for n in range(1, n_max + 1):
            vec_gen, vec_ref = tfidf_vec(gen_words, refs, n)
            # Cosine similarity
            dot = sum(vec_gen.get(ng, 0) * vec_ref.get(ng, 0) for ng in vec_ref)
            norm_gen = _math.sqrt(sum(v**2 for v in vec_gen.values())) + 1e-10
            norm_ref = _math.sqrt(sum(v**2 for v in vec_ref.values())) + 1e-10
            # Facteur de brièveté (pénalise les captions trop courtes)
            if len(gen_words) < len(refs[0]) if refs else 0:
                bp = _math.exp(1 - len(refs[0]) / (len(gen_words) + 1e-10))
            else:
                bp = 1.0
            score_n.append(bp * dot / (norm_gen * norm_ref))
        scores.append(sum(score_n) / n_max)

    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Classe pour gérer l'entraînement du modèle.
    """

    def __init__(self, model, train_loader, val_loader, vocabulary, config):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.vocabulary   = vocabulary
        self.config       = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de : {self.device}")

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocabulary.word2idx[vocabulary.pad_token],
            label_smoothing=0.1   # Lisse les targets → évite la sur-confiance
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )

        self.train_losses  = []
        self.val_losses    = []
        self.best_val_loss = float('inf')

        # Métriques linguistiques suivies à chaque epoch (ou tous les N)
        self.perplexities  = []          # exp(val_loss)
        self.bleu1_scores  = []          # [(epoch, score), ...]
        self.bleu4_scores  = []
        self.meteor_scores = []          # [(epoch, score), ...]
        self.cider_scores  = []          # [(epoch, score), ...]

        self.best_bleu4   = 0.0
        self.best_meteor  = 0.0
        self.best_cider   = 0.0

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'],        exist_ok=True)

    # -------------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()
        total_loss  = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')

        for images, captions, lengths in pbar:
            images   = images.to(self.device)
            captions = captions.to(self.device)

            inputs  = captions[:, :-1]   # Tout sauf <END>
            targets = captions[:, 1:]    # Tout sauf <START>

            outputs = self.model(images, inputs)        # (B, T, vocab)
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)

            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    # -------------------------------------------------------------------------

    def validate(self):
        self.model.eval()
        total_loss  = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader,
                                                   desc='Validation'):
                images   = images.to(self.device)
                captions = captions.to(self.device)

                inputs  = captions[:, :-1]
                targets = captions[:, 1:]

                outputs = self.model(images, inputs)
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / num_batches

    # -------------------------------------------------------------------------

    def _collect_predictions(self, num_samples):
        """
        Génère des captions pour `num_samples` images de la validation.

        Retourne deux listes parallèles :
          - generated_list : list[list[str]]  — mots générés par image
          - reference_list : list[list[list[str]]] — liste de références par image
            (une seule référence ici car le DataLoader ne fournit qu'une caption
            par image à la fois ; CIDEr est plus précis avec plusieurs références
            officielles, mais cela nécessiterait un DataLoader dédié)
        """
        self.model.eval()
        generated_list = []
        reference_list = []
        count = 0

        start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]
        pad_token   = self.vocabulary.word2idx[self.vocabulary.pad_token]

        with torch.no_grad():
            for images, captions, lengths in self.val_loader:
                if count >= num_samples:
                    break
                images = images.to(self.device)

                for i in range(images.size(0)):
                    if count >= num_samples:
                        break

                    # Génération greedy
                    features = self.model.encoder(images[i:i+1])
                    generated = self.model.decoder.generate(
                        features,
                        max_length=self.config.get('max_caption_length', 20),
                        start_token=start_token,
                        end_token=end_token
                    )

                    # Décodage tokens → mots (filtre tokens spéciaux)
                    gen_ids = [
                        t.item() if torch.is_tensor(t) else t
                        for t in generated[0]
                        if (t.item() if torch.is_tensor(t) else t)
                        not in [start_token, end_token, pad_token]
                    ]
                    ref_ids = [
                        t.item() for t in captions[i]
                        if t.item() not in [start_token, end_token, pad_token]
                    ]

                    gen_words = self.vocabulary.denumericalize(gen_ids).split()
                    ref_words = self.vocabulary.denumericalize(ref_ids).split()

                    if gen_words and ref_words:
                        generated_list.append(gen_words)
                        reference_list.append([ref_words])
                        count += 1

        return generated_list, reference_list

    # -------------------------------------------------------------------------

    def compute_bleu(self, generated_list, reference_list):
        """
        Calcule BLEU-1 et BLEU-4 à partir des listes déjà collectées.

        - BLEU-1 : précision sur les mots individuels (0→1)
        - BLEU-4 : précision sur les séquences de 4 mots (plus exigeant)
        """
        if not BLEU_AVAILABLE or not generated_list:
            return None, None

        smooth = SmoothingFunction().method1
        bleu1 = corpus_bleu(reference_list, generated_list,
                            weights=(1, 0, 0, 0),
                            smoothing_function=smooth)
        bleu4 = corpus_bleu(reference_list, generated_list,
                            weights=(.25, .25, .25, .25),
                            smoothing_function=smooth)
        return bleu1, bleu4

    def compute_meteor(self, generated_list, reference_list):
        """
        Calcule le score METEOR moyen sur le corpus.

        METEOR (Metric for Evaluation of Translation with Explicit ORdering)
        aligne les mots de la caption générée avec la référence en tenant compte
        des synonymes (WordNet) et des formes fléchies (stemming).
        Plus robuste que BLEU pour capturer la paraphrase.
        Valeur : 0→1, typiquement 0.20–0.35 pour un bon modèle.
        """
        if not METEOR_AVAILABLE or not generated_list:
            return None

        scores = []
        for gen_words, refs in zip(generated_list, reference_list):
            # meteor_score attend des strings, pas des listes de mots
            gen_str  = ' '.join(gen_words)
            ref_strs = [' '.join(r) for r in refs]
            # Prend le max sur les références disponibles
            s = max(meteor_score([ref], gen_str) for ref in ref_strs)
            scores.append(s)

        return sum(scores) / len(scores) if scores else None

    def compute_cider(self, generated_list, reference_list):
        """
        Calcule le score CIDEr moyen sur le corpus.

        CIDEr (Consensus-based Image Description Evaluation) pondère les
        n-grammes par leur rareté dans le corpus (TF-IDF), ce qui lui permet
        de valoriser les descriptions précises plutôt que les mots génériques.
        Valeur : ≥ 0, typiquement 0.5–1.2 pour un bon modèle.
        """
        if not generated_list:
            return None
        return compute_cider_score(generated_list, reference_list)

    # -------------------------------------------------------------------------

    def train(self):
        print("\n" + "="*70)
        print("DÉBUT DE L'ENTRAÎNEMENT (COCO)")
        print("="*70)

        params = self.model.get_num_params()
        print(f"\nNombre de paramètres : {params['total']:,}")
        print(f"  Encoder : {params['encoder']:,}")
        print(f"  Decoder : {params['decoder']:,}")

        print(f"\nConfiguration :")
        print(f"  Encoder type  : {self.config['encoder_type']}")
        print(f"  Epochs        : {self.config['num_epochs']}")
        print(f"  Batch size    : {self.config['batch_size']}")
        print(f"  Learning rate : {self.config['learning_rate']}")
        print(f"  Device        : {self.device}")

        bleu_every = self.config.get('bleu_every', 2)

        start_time       = time.time()
        patience_counter = 0

        for epoch in range(self.config['num_epochs']):

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            # ── Perplexité ─────────────────────────────────────────────────
            ppl = math.exp(val_loss)
            self.perplexities.append(ppl)

            # ── BLEU / METEOR / CIDEr tous les N epochs ────────────────────
            bleu1 = bleu4 = meteor = cider = None

            if (epoch + 1) % bleu_every == 0:
                num_samples = self.config.get('bleu_num_samples', 500)
                generated_list, reference_list = self._collect_predictions(num_samples)

                if generated_list:
                    # BLEU
                    bleu1, bleu4 = self.compute_bleu(generated_list, reference_list)

                    # METEOR
                    meteor = self.compute_meteor(generated_list, reference_list)

                    # CIDEr
                    cider = self.compute_cider(generated_list, reference_list)

                    # Enregistrement
                    ep = epoch + 1
                    if bleu1 is not None:
                        self.bleu1_scores.append((ep, bleu1))
                        self.bleu4_scores.append((ep, bleu4))
                        # BUG CORRIGÉ : comparer avant la mise à jour du best
                        is_best_bleu4 = bleu4 > self.best_bleu4
                        if is_best_bleu4:
                            self.best_bleu4 = bleu4
                    if meteor is not None:
                        is_best_meteor = meteor > self.best_meteor
                        self.meteor_scores.append((ep, meteor))
                        if is_best_meteor:
                            self.best_meteor = meteor
                    if cider is not None:
                        is_best_cider = cider > self.best_cider
                        self.cider_scores.append((ep, cider))
                        if is_best_cider:
                            self.best_cider = cider

            # ── Affichage ──────────────────────────────────────────────────
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  Train Loss  : {train_loss:.4f}")
            print(f"  Val Loss    : {val_loss:.4f}  │  PPL : {ppl:.2f}")
            if bleu1 is not None:
                print(f"  BLEU-1      : {bleu1:.4f}")
                print(f"  BLEU-4      : {bleu4:.4f}"
                      + ("  ★ best" if bleu4 >= self.best_bleu4 else ""))
            if meteor is not None:
                print(f"  METEOR      : {meteor:.4f}"
                      + ("  ★ best" if meteor >= self.best_meteor else ""))
            if cider is not None:
                print(f"  CIDEr       : {cider:.4f}"
                      + ("  ★ best" if cider >= self.best_cider else ""))
            if bleu1 is None and BLEU_AVAILABLE:
                next_bleu = bleu_every - ((epoch + 1) % bleu_every)
                print(f"  Métriques   : (prochain calcul dans {next_bleu} "
                      f"epoch{'s' if next_bleu > 1 else ''})")

            # ── Checkpoint régulier ────────────────────────────────────────
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                ckpt_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                caption_model2.save_model(
                    self.model, ckpt_path,
                    optimizer=self.optimizer,
                    epoch=epoch, loss=val_loss,
                    vocab=self.vocabulary
                )

            # ── Meilleur modèle (critère : val loss) ──────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config['checkpoint_dir'],
                                         'best_model.pth')
                caption_model2.save_model(
                    self.model, best_path,
                    optimizer=self.optimizer,
                    epoch=epoch, loss=val_loss,
                    vocab=self.vocabulary
                )
                patience_counter = 0
                print(f"  ✓ Nouveau meilleur modèle (loss : {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ✗ Pas d'amélioration "
                      f"(patience : {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config['patience']:
                    print("\nEarly stopping déclenché !")
                    break

        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/60:.2f} minutes")
        print(f"Meilleure val loss  : {self.best_val_loss:.4f}  │  "
              f"PPL : {math.exp(self.best_val_loss):.2f}")
        if self.best_bleu4  > 0: print(f"Meilleur BLEU-4     : {self.best_bleu4:.4f}")
        if self.best_meteor > 0: print(f"Meilleur METEOR     : {self.best_meteor:.4f}")
        if self.best_cider  > 0: print(f"Meilleur CIDEr      : {self.best_cider:.4f}")

        self.plot_learning_curves()
        self.save_history()

    # -------------------------------------------------------------------------

    def plot_learning_curves(self):
        """
        Génère une figure à 5 panneaux maximum :
          1. Loss train vs val
          2. Perplexité val
          3. BLEU-1 et BLEU-4
          4. METEOR
          5. CIDEr
        Les panneaux 3/4/5 ne sont affichés que si des données sont disponibles.
        """
        epochs_all = range(1, len(self.train_losses) + 1)
        has_bleu   = len(self.bleu1_scores)  > 0
        has_meteor = len(self.meteor_scores) > 0
        has_cider  = len(self.cider_scores)  > 0

        n_panels = 2 + int(has_bleu) + int(has_meteor) + int(has_cider)
        # Minimum 2 panneaux (loss + PPL), largeur adaptée
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        # Garantir que axes est toujours indexable même avec n_panels=1
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(
            f'Entraînement COCO — {self.config["encoder_type"]}',
            fontsize=14, fontweight='bold', y=1.02
        )

        panel = 0  # index courant du panneau

        # ── Panneau 1 : Loss ────────────────────────────────────────────────
        ax = axes[panel]; panel += 1
        ax.plot(epochs_all, self.train_losses, 'b-',  label='Train Loss', linewidth=2)
        ax.plot(epochs_all, self.val_losses,   'r-',  label='Val Loss',   linewidth=2)
        best_epoch = self.val_losses.index(min(self.val_losses)) + 1
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                   label=f'Best (ep.{best_epoch})')
        ax.set_title('Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy Loss')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # ── Panneau 2 : Perplexité ───────────────────────────────────────────
        ax = axes[panel]; panel += 1
        ax.plot(epochs_all, self.perplexities, color='purple', linewidth=2)
        ax.set_title('Perplexité (exp(val_loss))', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('PPL')
        ax.grid(True, alpha=0.3)
        ax.annotate(f'PPL={self.perplexities[0]:.1f}',
                    xy=(1, self.perplexities[0]), fontsize=8, color='gray')
        best_ppl_idx = self.perplexities.index(min(self.perplexities))
        ax.annotate(f'PPL={min(self.perplexities):.1f} (best)',
                    xy=(best_ppl_idx + 1, min(self.perplexities)),
                    fontsize=8, color='green',
                    xytext=(5, 10), textcoords='offset points')

        # ── Panneau 3 : BLEU ─────────────────────────────────────────────────
        if has_bleu:
            ax = axes[panel]; panel += 1
            bleu_epochs = [e for e, _ in self.bleu1_scores]
            bleu1_vals  = [s for _, s in self.bleu1_scores]
            bleu4_vals  = [s for _, s in self.bleu4_scores]
            ax.plot(bleu_epochs, bleu1_vals, 'b-o', label='BLEU-1', linewidth=2, markersize=5)
            ax.plot(bleu_epochs, bleu4_vals, 'r-o', label='BLEU-4', linewidth=2, markersize=5)
            ax.set_title('BLEU', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
            ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(max(bleu1_vals) * 1.2, 0.1))
            best_b4_idx = bleu4_vals.index(max(bleu4_vals))
            ax.annotate(f'Best BLEU-4\n{max(bleu4_vals):.4f}',
                        xy=(bleu_epochs[best_b4_idx], bleu4_vals[best_b4_idx]),
                        fontsize=8, color='red',
                        xytext=(8, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))

        # ── Panneau 4 : METEOR ───────────────────────────────────────────────
        if has_meteor:
            ax = axes[panel]; panel += 1
            meteor_epochs = [e for e, _ in self.meteor_scores]
            meteor_vals   = [s for _, s in self.meteor_scores]
            ax.plot(meteor_epochs, meteor_vals, 'g-o', label='METEOR',
                    linewidth=2, markersize=5)
            ax.set_title('METEOR', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
            ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(max(meteor_vals) * 1.2, 0.1))
            best_m_idx = meteor_vals.index(max(meteor_vals))
            ax.annotate(f'Best\n{max(meteor_vals):.4f}',
                        xy=(meteor_epochs[best_m_idx], meteor_vals[best_m_idx]),
                        fontsize=8, color='green',
                        xytext=(8, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green', lw=1))

        # ── Panneau 5 : CIDEr ────────────────────────────────────────────────
        if has_cider:
            ax = axes[panel]; panel += 1
            cider_epochs = [e for e, _ in self.cider_scores]
            cider_vals   = [s for _, s in self.cider_scores]
            ax.plot(cider_epochs, cider_vals, color='darkorange', marker='o',
                    linewidth=2, markersize=5, label='CIDEr')
            ax.set_title('CIDEr', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
            ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(max(cider_vals) * 1.2, 0.1))
            best_c_idx = cider_vals.index(max(cider_vals))
            ax.annotate(f'Best\n{max(cider_vals):.4f}',
                        xy=(cider_epochs[best_c_idx], cider_vals[best_c_idx]),
                        fontsize=8, color='darkorange',
                        xytext=(8, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1))

        plt.tight_layout()
        save_path = os.path.join(self.config['log_dir'], 'learning_curves_coco.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nCourbes sauvegardées → {save_path}")
        plt.close()

    def save_history(self):
        history = {
            'train_losses':  self.train_losses,
            'val_losses':    self.val_losses,
            'perplexities':  self.perplexities,
            'bleu1_scores':  self.bleu1_scores,
            'bleu4_scores':  self.bleu4_scores,
            'meteor_scores': self.meteor_scores,
            'cider_scores':  self.cider_scores,
            'best_val_loss': self.best_val_loss,
            'best_bleu4':    self.best_bleu4,
            'best_meteor':   self.best_meteor,
            'best_cider':    self.best_cider,
            'config':        self.config
        }
        save_path = os.path.join(self.config['log_dir'],
                                 'training_history_coco.json')
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Historique sauvegardé → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("PRÉPARATION DES DONNÉES COCO")
    print("="*70)

    # ── Vocabulaire ────────────────────────────────────────────────────────────
    if os.path.exists(CONFIG['vocab_path']):
        print(f"\nChargement du vocabulaire depuis {CONFIG['vocab_path']}")
        vocab = vocabulary.Vocabulary.load(CONFIG['vocab_path'])
    else:
        print("\nConstruction du vocabulaire depuis le train set COCO...")
        train_caption_prep = CaptionPreprocessor(
            CONFIG['train_captions_file'],
            CONFIG['train_images_dir']
        )
        vocab = vocabulary.Vocabulary(freq_threshold=CONFIG['freq_threshold'])
        vocab.build_vocabulary(train_caption_prep.get_all_captions())
        vocab.save(CONFIG['vocab_path'])

    print(f"Taille du vocabulaire : {len(vocab)}")

    # ── Données — splits officiels COCO ────────────────────────────────────────
    print("\nChargement des paires train (COCO train2017)...")
    train_caption_prep = CaptionPreprocessor(
        CONFIG['train_captions_file'],
        CONFIG['train_images_dir']
    )
    train_pairs = train_caption_prep.get_image_caption_pairs()

    print("\nChargement des paires val (COCO val2017)...")
    val_caption_prep = CaptionPreprocessor(
        CONFIG['val_captions_file'],
        CONFIG['val_images_dir']
    )
    val_pairs = val_caption_prep.get_image_caption_pairs()

    # ── Transforms ─────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_prep = ImagePreprocessor(
        image_size=CONFIG['image_size'],
        normalize=False,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_loader, val_loader = data_loader.get_data_loaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        vocabulary=vocab,
        image_preprocessor=image_prep,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        shuffle_train=True
    )

    # ── Modèle ─────────────────────────────────────────────────────────────────
    print(f"\nCréation du modèle (encoder_type='{CONFIG['encoder_type']}')...")
    model = caption_model2.create_model(
        vocab_size    = len(vocab),
        embedding_dim = CONFIG['embedding_dim'],
        hidden_dim    = CONFIG['hidden_dim'],
        feature_dim   = CONFIG['feature_dim'],
        num_layers    = CONFIG['num_layers'],
        dropout       = CONFIG['dropout'],
        encoder_type  = CONFIG['encoder_type'],
        attention_dim = CONFIG.get('attention_dim', 256),
    )

    # ── Entraînement ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocab,
        config=CONFIG
    )

    trainer.train()

    print("\n" + "="*70)
    print("ENTRAÎNEMENT COCO TERMINÉ !")
    print("="*70)


if __name__ == "__main__":
    main()