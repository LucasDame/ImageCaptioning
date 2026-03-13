"""
Script d'entraînement COCO pour Image Captioning
=================================================

Compatible avec les trois encoder_type du nouveau modèle :
  'lite'      → EncoderCNNLite  + DecoderLSTM
  'full'      → EncoderCNN      + DecoderLSTM  (résiduel)
  'attention' → EncoderSpatial  + DecoderWithAttention

Correctifs v3 :
  - Scheduler : ReduceLROnPlateau (patience=2) → CosineAnnealingWarmRestarts
                (T_0=10, T_mult=2). Le LR ne s'effondre plus après epoch 30.
  - Régularisation doubly stochastic : pénalise l'attention qui se concentre
    sur les coins au lieu de couvrir toute l'image (uniquement encoder_type='attention').
  - bleu_num_samples porté à 2000 par défaut pour un CIDEr stable.
  - Early stopping basé sur le CIDEr (si disponible) plutôt que sur la val loss
    seule, ce qui évite de stopper un modèle dont le CIDEr progresse encore.
  - Patience early stopping portée à 10 pour laisser le scheduler compléter
    ses cycles.
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
CIDER_AVAILABLE = True  # implémentation interne, toujours disponible


from utils import vocabulary, data_loader
from utils.preprocessing_coco import CaptionPreprocessor, ImagePreprocessor
from models2 import caption_model2

from config_coco2 import CONFIG


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
            idf[(n, ng)] = _math.log((num_refs + 1.0) / (df + 1.0))

    def tfidf_vec(words, refs_for_image, n):
        tf_gen = _ngrams(words, n)
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
            dot = sum(vec_gen.get(ng, 0) * vec_ref.get(ng, 0) for ng in vec_ref)
            norm_gen = _math.sqrt(sum(v**2 for v in vec_gen.values())) + 1e-10
            norm_ref = _math.sqrt(sum(v**2 for v in vec_ref.values())) + 1e-10
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

    def __init__(self, model, train_loader, val_loader, vocabulary, config,
                 val_pairs=None):
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
            label_smoothing=0.1
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )

        # ── CosineAnnealingWarmRestarts ────────────────────────────────────
        # Remplace ReduceLROnPlateau(patience=2) qui tuait le LR trop tôt.
        #
        # Fonctionnement :
        #   T_0    = 10  → premier cycle de 10 epochs après le warmup
        #   T_mult = 2   → cycles suivants : 20, 40, 80 epochs...
        #   Le LR remonte au début de chaque cycle → le modèle peut
        #   explorer de nouveaux bassins de convergence.
        #
        # Pourquoi pas ReduceLROnPlateau ?
        #   Avec patience=2, le scheduler réduisait le LR dès l'epoch 8
        #   (warmup fini à 5 + 2 epochs sans amélioration = 7).
        #   À l'epoch ~28 le LR était à 7.5e-5, à l'epoch ~48 à 3.75e-5,
        #   à l'epoch ~58 au minimum 1e-5 → modèle gelé, CIDEr stagnant.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('cosine_T0', 10),
            T_mult=config.get('cosine_T_mult', 2),
            eta_min=config.get('lr_min', 1e-5),
        )
        # Flag pour réinitialiser base_lrs UNE SEULE FOIS après le warmup
        # (bug : le scheduler est créé avec LR=lr/N au lieu de lr_target)
        self._scheduler_initialized = False

        # Indique si le decoder supporte la régularisation d'attention
        self._use_attention_reg = (
            config.get('encoder_type') == 'attention'
            and config.get('attention_lambda', 1.0) > 0.0
        )
        self._attention_lambda = config.get('attention_lambda', 1.0)

        # ── val_refs : image_path → [ref1, ref2, ...] (jusqu'à 5/image) ────
        # Construit depuis val_pairs pour que CIDEr dispose des 5 références
        # officielles COCO. Sans ce dict, _collect_predictions n'a qu'1 ref/image
        # → IDF faux → CIDEr ~0.07 au lieu de ~0.5-0.7.
        self._val_refs       = {}
        self._val_image_order = []
        if val_pairs:
            for pair in val_pairs:
                path  = pair['image_path']
                words = [w for w in pair['caption'].lower().split()
                         if w not in {'', '.', ',', '!', '?'}]
                if words:
                    if path not in self._val_refs:
                        self._val_refs[path] = []
                        self._val_image_order.append(path)
                    self._val_refs[path].append(words)
            print(f"val_refs : {len(self._val_refs)} images, "
                  f"{sum(len(v) for v in self._val_refs.values())} captions")
        else:
            print("⚠️  val_pairs non fourni → CIDEr utilisera 1 ref/image")

        self.train_losses  = []
        self.val_losses    = []
        self.best_val_loss = float('inf')

        self.perplexities  = []
        self.bleu1_scores  = []
        self.bleu4_scores  = []
        self.meteor_scores = []
        self.cider_scores  = []

        self.best_bleu4   = 0.0
        self.best_meteor  = 0.0
        self.best_cider   = 0.0

        self.lr_history = []

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

            # ── CORRECTIF 2 : passe forward avec collecte des alphas ──────
            # Pour encoder_type='attention', on demande au decoder de
            # retourner les poids d'attention à chaque pas afin d'appliquer
            # la régularisation doubly stochastic.
            if self._use_attention_reg:
                outputs, alphas = self.model.forward_with_alphas(images, inputs)
            else:
                outputs = self.model(images, inputs)
                alphas  = None

            outputs_flat = outputs.reshape(-1, outputs.shape[2])
            targets_flat = targets.reshape(-1)

            loss = self.criterion(outputs_flat, targets_flat)

            # ── CORRECTIF 2 (suite) : régularisation doubly stochastic ───
            # Principe (Show, Attend and Tell, Xu et al. 2015, §4.2.1) :
            #   On veut que la somme des poids d'attention sur tous les pas
            #   soit proche de 1 pour chaque région spatiale.
            #   Si alpha[t, p] est le poids de la région p au pas t, alors
            #   idéalement Σ_t alpha[t, p] ≈ 1  pour tout p.
            #   Pénalité : λ · mean((1 - Σ_t alpha[:, p])²)
            #
            # Effet concret : empêche le modèle de toujours regarder les
            # mêmes coins de la grille 7×7, force une couverture uniforme.
            if alphas is not None:
                # alphas : (B, T, P) avec P = 49 pour une grille 7×7
                attention_sum = alphas.sum(dim=1)                 # (B, P)
                attention_reg = ((1.0 - attention_sum) ** 2).mean()
                loss = loss + self._attention_lambda * attention_reg

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

                # Pas de régularisation en validation (pas de backward)
                outputs = self.model(images, inputs)
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / num_batches

    # -------------------------------------------------------------------------

    def _collect_predictions(self, num_samples):
        """
        Génère des captions pour num_samples images et retourne les références.

        val_loader a shuffle=False → dataset.pairs[global_sample] correspond
        exactement à l'entrée courante. On lit image_path depuis pairs[idx]
        pour retrouver les 5 refs dans self._val_refs.
        seen_paths évite de générer deux fois la même image (le val loader
        expose chaque image 5 fois, une par caption).
        """
        self.model.eval()

        start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]
        pad_token   = self.vocabulary.word2idx[self.vocabulary.pad_token]

        generated_list = []
        reference_list = []
        seen_paths     = set()
        dataset        = self.val_loader.dataset
        global_sample  = 0

        with torch.no_grad():
            for images, captions, lengths in self.val_loader:
                if len(generated_list) >= num_samples:
                    break
                images_gpu = images.to(self.device)

                for i in range(images.size(0)):
                    if len(generated_list) >= num_samples:
                        break

                    # Chemin image depuis le dataset (shuffle=False garanti)
                    try:
                        image_path = dataset.pairs[global_sample]['image_path']
                    except (AttributeError, IndexError, KeyError):
                        image_path = None
                    global_sample += 1

                    # Sauter les doublons (même image, caption différente)
                    if image_path in seen_paths:
                        continue

                    # Références : 5/image depuis _val_refs, ou 1 en fallback
                    if image_path and image_path in self._val_refs:
                        refs_for_image = self._val_refs[image_path]
                    else:
                        ref_ids   = [t.item() for t in captions[i]
                                     if t.item() not in [start_token, end_token, pad_token]]
                        ref_words = self.vocabulary.denumericalize(ref_ids).split()
                        refs_for_image = [ref_words] if ref_words else None

                    if not refs_for_image:
                        continue

                    # Génération greedy
                    features  = self.model.encoder(images_gpu[i:i+1])
                    generated = self.model.decoder.generate(
                        features,
                        max_length=self.config.get('max_caption_length', 20),
                        start_token=start_token,
                        end_token=end_token
                    )
                    gen_ids   = [
                        t.item() if torch.is_tensor(t) else t
                        for t in generated[0]
                        if (t.item() if torch.is_tensor(t) else t)
                        not in [start_token, end_token, pad_token]
                    ]
                    gen_words = self.vocabulary.denumericalize(gen_ids).split()

                    if gen_words:
                        generated_list.append(gen_words)
                        reference_list.append(refs_for_image)
                        if image_path:
                            seen_paths.add(image_path)

        return generated_list, reference_list

    # -------------------------------------------------------------------------

    def compute_bleu(self, generated_list, reference_list):
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
        if not METEOR_AVAILABLE or not generated_list:
            return None

        scores = []
        for gen_words, refs in zip(generated_list, reference_list):
            s = meteor_score(refs, gen_words)
            scores.append(s)

        return sum(scores) / len(scores) if scores else None

    def compute_cider(self, generated_list, reference_list):
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
        print(f"  Encoder type        : {self.config['encoder_type']}")
        print(f"  Epochs              : {self.config['num_epochs']}")
        print(f"  Batch size          : {self.config['batch_size']}")
        print(f"  Learning rate       : {self.config['learning_rate']}")
        print(f"  Cosine T0/T_mult    : {self.config.get('cosine_T0',10)}/{self.config.get('cosine_T_mult',2)}")
        print(f"  Attention lambda    : {self.config.get('attention_lambda', 1.0)}")
        print(f"  bleu_num_samples    : {self.config.get('bleu_num_samples', 2000)}")
        print(f"  Device              : {self.device}")

        bleu_every    = self.config.get('bleu_every', 1)
        warmup_epochs = self.config.get('warmup_epochs', 5)

        start_time       = time.time()
        patience_counter = 0

        # ── Initialisation warmup ─────────────────────────────────────────────
        # On monte le LR linéairement de lr/N jusqu'à lr_target pendant le warmup.
        # Le CosineAnnealingWarmRestarts prend le relais à partir de epoch warmup_epochs.
        _lr_target = self.config['learning_rate']
        _set_lr    = lambda new_lr: [
            pg.update({'lr': new_lr}) for pg in self.optimizer.param_groups
        ]
        _get_lr    = lambda: self.optimizer.param_groups[0]['lr']

        _set_lr(_lr_target / max(warmup_epochs, 1))

        for epoch in range(self.config['num_epochs']):

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # ── Gestion du LR ─────────────────────────────────────────────────
            lr_before = _get_lr()

            if epoch < warmup_epochs:
                # Montée linéaire pendant le warmup
                new_lr = _lr_target * (epoch + 1) / warmup_epochs
                _set_lr(new_lr)
                lr_note = f"warmup {epoch+1}/{warmup_epochs}"
            else:
                # BUG CORRIGÉ : le scheduler a été créé quand le LR optimizer
                # valait lr/warmup_epochs (pas lr_target). Il stocke cette valeur
                # dans base_lrs → son cosine descend de lr/N à eta_min au lieu
                # de lr_target à eta_min. On corrige base_lrs une seule fois.
                if not self._scheduler_initialized:
                    self.scheduler.base_lrs = [_lr_target] * len(
                        self.scheduler.base_lrs
                    )
                    self._scheduler_initialized = True

                cosine_epoch = epoch - warmup_epochs
                self.scheduler.step(cosine_epoch)
                lr_note = "CosineAnnealingWarmRestarts"

            lr_after   = _get_lr()
            lr_changed = lr_after < lr_before * 0.99
            self.lr_history.append((epoch + 1, lr_after))

            # ── Perplexité ─────────────────────────────────────────────────
            ppl = math.exp(val_loss)
            self.perplexities.append(ppl)

            # ── BLEU / METEOR / CIDEr tous les N epochs ────────────────────
            bleu1 = bleu4 = meteor = cider = None

            if (epoch + 1) % bleu_every == 0:
                # CORRECTIF 3 : bleu_num_samples augmenté à 2000 (config)
                # pour un IDF corpus suffisamment grand → CIDEr stable
                num_samples = self.config.get('bleu_num_samples', 2000)
                generated_list, reference_list = self._collect_predictions(num_samples)

                if generated_list:
                    bleu1, bleu4 = self.compute_bleu(generated_list, reference_list)
                    meteor = self.compute_meteor(generated_list, reference_list)
                    cider  = self.compute_cider(generated_list, reference_list)

                    ep = epoch + 1
                    if bleu1 is not None:
                        self.bleu1_scores.append((ep, bleu1))
                        self.bleu4_scores.append((ep, bleu4))
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
            _lr_tag = ""
            if epoch >= warmup_epochs and lr_changed:
                _lr_tag = "  ↓ cosine reset"
            elif epoch < warmup_epochs:
                _lr_tag = f"  ↑ {lr_note}"
            print(f"  LR          : {lr_after:.2e}{_lr_tag}")
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
                print(f"  Métriques   : (prochain dans {next_bleu} "
                      f"epoch{'s' if next_bleu > 1 else ''})")

            # ── Plot PNG + JSON ────────────────────────────────────────────
            if (epoch + 1) % bleu_every == 0:
                self.plot_learning_curves()
                self.save_history()

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

            # ── Meilleur modèle (val loss) ─────────────────────────────────
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
                print(f"  ✓ Nouveau meilleur modèle (loss : {val_loss:.4f})")

            # ── Meilleur modèle CIDEr (sauvegarde séparée) ────────────────
            if cider is not None and cider >= self.best_cider:
                best_cider_path = os.path.join(self.config['checkpoint_dir'],
                                               'best_model_cider.pth')
                caption_model2.save_model(
                    self.model, best_cider_path,
                    optimizer=self.optimizer,
                    epoch=epoch, loss=val_loss,
                    vocab=self.vocabulary
                )
                print(f"  ✓ Nouveau meilleur modèle CIDEr ({cider:.4f})")

            # ── CORRECTIF 4 : Early stopping basé sur CIDEr ou val loss ───
            # Si le CIDEr est disponible, on l'utilise comme critère principal
            # car c'est la métrique qui reflète le mieux la qualité des captions.
            # Si le CIDEr n'est pas encore calculé (bleu_every > 1), on se
            # rabat sur la val loss.
            #
            # La patience est portée à 10 (config) pour laisser au scheduler
            # le temps de compléter au moins un cycle complet (T_0=10).
            if cider is not None:
                # Le compteur de patience est remis à zéro si le CIDEr s'améliore
                if cider >= self.best_cider:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  ✗ CIDEr sans amélioration "
                          f"(patience : {patience_counter}/{self.config['patience']})")
            else:
                # Pas encore de CIDEr : on surveille la val loss
                if val_loss < self.best_val_loss + 1e-4:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  ✗ Val loss sans amélioration "
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
        epochs_all = range(1, len(self.train_losses) + 1)
        has_bleu   = len(self.bleu1_scores)  > 0
        has_meteor = len(self.meteor_scores) > 0
        has_cider  = len(self.cider_scores)  > 0

        n_panels = 3 + int(has_bleu) + int(has_meteor) + int(has_cider)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(
            f'Entraînement COCO — {self.config["encoder_type"]}',
            fontsize=14, fontweight='bold', y=1.02
        )

        panel = 0

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
        best_ppl_idx = self.perplexities.index(min(self.perplexities))
        ax.annotate(f'PPL={min(self.perplexities):.1f} (best)',
                    xy=(best_ppl_idx + 1, min(self.perplexities)),
                    fontsize=8, color='green',
                    xytext=(5, 10), textcoords='offset points')

        # ── Panneau 3 : Learning Rate ────────────────────────────────────────
        ax = axes[panel]; panel += 1
        if self.lr_history:
            lr_epochs = [e for e, _ in self.lr_history]
            lr_vals   = [lr for _, lr in self.lr_history]
            ax.plot(lr_epochs, lr_vals, color='teal', linewidth=2)
            ax.set_yscale('log')

            _wu = self.config.get('warmup_epochs', 5)
            if _wu > 0 and max(lr_epochs) >= 1:
                ax.axvspan(1, min(_wu, max(lr_epochs)),
                           alpha=0.10, color='orange',
                           label=f'Warmup ({_wu} ep.)')

            # Marqueurs aux redémarrages du cosine (T_0, T_0+T_1, ...)
            T0     = self.config.get('cosine_T0', 10)
            T_mult = self.config.get('cosine_T_mult', 2)
            t      = T0
            restart_ep = _wu + t
            while restart_ep <= max(lr_epochs):
                ax.axvline(x=restart_ep, color='steelblue',
                           linestyle=':', alpha=0.6, linewidth=1.2,
                           label='Cosine restart' if t == T0 else '')
                t          *= T_mult
                restart_ep  = _wu + t

        ax.set_title('Learning Rate (Cosine + Warmup)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('LR (log scale)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

        # ── Panneau 4 : BLEU ─────────────────────────────────────────────────
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

        # ── Panneau 5 : METEOR ───────────────────────────────────────────────
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

        # ── Panneau 6 : CIDEr ────────────────────────────────────────────────
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
        save_path = os.path.join(self.config['log_dir'], 'learning_curves_coco2.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nCourbes sauvegardées → {save_path}")
        plt.close()

    def save_history(self):
        history = {
            'train_losses':  self.train_losses,
            'val_losses':    self.val_losses,
            'perplexities':  self.perplexities,
            'lr_history':    self.lr_history,
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
                                 'training_history_coco2.json')
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
        config=CONFIG,
        val_pairs=val_pairs        # ← 5 refs/image pour CIDEr correct
    )

    trainer.train()

    print("\n" + "="*70)
    print("ENTRAÎNEMENT COCO TERMINÉ !")
    print("="*70)


if __name__ == "__main__":
    main()