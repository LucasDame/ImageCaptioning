"""
train.py — Entraînement Image Captioning COCO
==============================================

Utilisation :
    python train.py --model densenet --scheduler cosine
    python train.py --model resnet   --scheduler plateau
    python train.py --model cnn      --scheduler cosine
    python train.py --model densenet --scheduler cosine --fast   # mode dev rapide
    python train.py --model densenet --scheduler cosine --resume checkpoints/densenet/best_model.pth

Architectures (--model) :
    cnn       → EncoderCNN     + DecoderLSTM            (résiduel from scratch, vecteur global)
    resnet    → EncoderSpatial + DecoderWithAttention   (résiduel from scratch + Bahdanau)
    densenet  → EncoderDenseNet + DecoderWithAttention  (DenseNet-121 from scratch + Bahdanau) ← recommandé

Schedulers (--scheduler) :
    plateau   → ReduceLROnPlateau(patience=10, factor=0.5)
                Early stop après `patience` epochs sans amélioration.
    cosine    → CosineAnnealingWarmRestarts(T0, T_mult)
                Sauvegarde le meilleur modèle du cycle à chaque fin de cycle.
                Vérifie l'amélioration (val loss) à chaque fin de cycle.
                Early stop après `max_no_improve_cycles` cycles sans amélioration.
                Moyenne des poids des meilleurs checkpoints de cycle en fin d'entraînement.
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# ── Métriques BLEU ────────────────────────────────────────────────────────────
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
    print("⚠️  NLTK non installé — BLEU désactivé.  pip install nltk")
    BLEU_AVAILABLE = False

# ── Métrique METEOR ───────────────────────────────────────────────────────────
try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False

from config import get_config
from models.caption_model import create_model, save_model, load_model
from utils.vocabulary import Vocabulary
from utils.preprocessing import CaptionPreprocessor, ImagePreprocessor
from utils.data_loader import get_data_loaders


# =============================================================================
# CIDEr INTERNE
# =============================================================================

def _ngrams(words, n):
    from collections import Counter
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def compute_cider_score(generated_list, reference_list, n_max=4):
    """CIDEr-D sur corpus complet."""
    from collections import Counter
    import math as _math

    num_refs = len(reference_list)
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
        from collections import Counter as C
        tf_gen = _ngrams(words, n)
        tf_ref = C()
        for ref in refs_for_image:
            for ng, cnt in _ngrams(ref, n).items():
                tf_ref[ng] += cnt
        if refs_for_image:
            for ng in tf_ref:
                tf_ref[ng] /= len(refs_for_image)
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
            dot = sum(vec_gen.get(ng, 0) * vec_ref.get(ng, 0) for ng in vec_ref)
            norm_gen = _math.sqrt(sum(v**2 for v in vec_gen.values())) + 1e-10
            norm_ref = _math.sqrt(sum(v**2 for v in vec_ref.values())) + 1e-10
            bp = 1.0
            if refs and len(gen_words) < len(refs[0]):
                bp = _math.exp(1 - len(refs[0]) / (len(gen_words) + 1e-10))
            score_n.append(bp * dot / (norm_gen * norm_ref))
        scores.append(sum(score_n) / n_max)

    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Entraîne le modèle d'image captioning avec le scheduler choisi.

    Schedulers :
      'plateau' : ReduceLROnPlateau avec patience=10.
                  Early stop si `patience` epochs sans amélioration.
      'cosine'  : CosineAnnealingWarmRestarts avec vérification d'amélioration
                  à chaque fin de cycle cosine.
                  Early stop si `max_no_improve_cycles` cycles sans amélioration.
    """

    def __init__(self, model, train_loader, val_loader, vocabulary, config,
                 scheduler_type='cosine', val_pairs=None):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.vocabulary     = vocabulary
        self.config         = config
        self.scheduler_type = scheduler_type

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device : {self.device}")
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

        # ── Scheduler ────────────────────────────────────────────────────────
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.get('plateau_patience', 10),
                factor=config.get('plateau_factor', 0.5),
                min_lr=config.get('lr_min', 1e-5),
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.get('cosine_T0', 10),
                T_mult=config.get('cosine_T_mult', 2),
                eta_min=config.get('lr_min', 1e-5),
            )
        else:
            raise ValueError(f"Scheduler inconnu : '{scheduler_type}'. Valeurs : 'plateau', 'cosine'.")

        self._cosine_initialized = False  # base_lrs à corriger après le warmup

        # ── Régularisation doubly stochastic ──────────────────────────────────
        arch = config.get('model', '')
        self._use_attention_reg = (
            arch in ('resnet', 'densenet')
            and config.get('attention_lambda', 1.0) > 0.0
        )
        self._attention_lambda = config.get('attention_lambda', 1.0)

        # ── Références val pour CIDEr (5 captions/image) ─────────────────────
        self._val_refs        = {}
        self._val_image_order = []

        if val_pairs:
            for pair in val_pairs:
                path    = pair['image_path']
                caption = pair['caption']
                words   = [w for w in caption.lower().split()
                           if w not in {'', '.', ',', '!', '?'}]
                if words:
                    if path not in self._val_refs:
                        self._val_refs[path] = []
                        self._val_image_order.append(path)
                    self._val_refs[path].append(words)
            print(f"val_refs : {len(self._val_refs)} images, "
                  f"{sum(len(v) for v in self._val_refs.values())} captions")
        else:
            print("⚠️  val_pairs non fourni → CIDEr avec 1 ref/image (sous-estimé)")

        # ── Historiques ───────────────────────────────────────────────────────
        self.train_losses  = []
        self.val_losses    = []
        self.perplexities  = []
        self.bleu1_scores  = []
        self.bleu4_scores  = []
        self.meteor_scores = []
        self.cider_scores  = []
        self.lr_history    = []

        self.best_val_loss = float('inf')
        self.best_bleu4    = 0.0
        self.best_meteor   = 0.0
        self.best_cider    = 0.0

        # ── Suivi des cycles cosine (pour sauvegarde et SWA) ─────────────────
        # best_cycle_val_loss : meilleure val loss vue dans le cycle courant
        # cycle_checkpoints   : liste des chemins sauvegardés à chaque fin de cycle
        # cycle_count         : numéro du cycle cosine courant (commence à 1)
        self._best_cycle_val_loss = float('inf')  # reset à chaque début de cycle
        self._cycle_checkpoints   = []             # chemins des best_model_cycle_N.pth
        self._cycle_count         = 0

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'],        exist_ok=True)

    # -------------------------------------------------------------------------
    # TRAIN EPOCH
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

            inputs  = captions[:, :-1]
            targets = captions[:, 1:]

            if self._use_attention_reg:
                outputs, alphas = self.model.forward_with_alphas(images, inputs)
            else:
                outputs = self.model(images, inputs)
                alphas  = None

            loss = self.criterion(
                outputs.reshape(-1, outputs.shape[2]),
                targets.reshape(-1)
            )

            if alphas is not None:
                attention_sum = alphas.sum(dim=1)
                loss = loss + self._attention_lambda * ((1.0 - attention_sum) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    # -------------------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------------------

    def validate(self):
        self.model.eval()
        total_loss  = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader, desc='Validation'):
                images   = images.to(self.device)
                captions = captions.to(self.device)
                inputs   = captions[:, :-1]
                targets  = captions[:, 1:]
                outputs  = self.model(images, inputs)
                loss     = self.criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    targets.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / num_batches

    # -------------------------------------------------------------------------
    # COLLECT PREDICTIONS
    # -------------------------------------------------------------------------

    def _collect_predictions(self, num_samples):
        """Génère des captions greedy sur num_samples images du val set."""
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

                    try:
                        image_path = dataset.pairs[global_sample]['image_path']
                    except (AttributeError, IndexError, KeyError):
                        image_path = None
                    global_sample += 1

                    if image_path in seen_paths:
                        continue

                    if image_path and image_path in self._val_refs:
                        refs_for_image = self._val_refs[image_path]
                    else:
                        ref_ids   = [t.item() for t in captions[i]
                                     if t.item() not in [start_token, end_token, pad_token]]
                        ref_words = self.vocabulary.denumericalize(ref_ids).split()
                        refs_for_image = [ref_words] if ref_words else None

                    if not refs_for_image:
                        continue

                    features  = self.model.encoder(images_gpu[i:i+1])
                    generated = self.model.decoder.generate(
                        features,
                        max_length=self.config.get('max_caption_length', 20),
                        start_token=start_token, end_token=end_token
                    )
                    gen_ids = [
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
    # MÉTRIQUES
    # -------------------------------------------------------------------------

    def compute_bleu(self, generated_list, reference_list):
        if not BLEU_AVAILABLE or not generated_list:
            return None, None
        smooth = SmoothingFunction().method1
        bleu1 = corpus_bleu(reference_list, generated_list,
                            weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu4 = corpus_bleu(reference_list, generated_list,
                            weights=(.25, .25, .25, .25), smoothing_function=smooth)
        return bleu1, bleu4

    def compute_meteor(self, generated_list, reference_list):
        if not METEOR_AVAILABLE or not generated_list:
            return None
        scores = [meteor_score(refs, gen) for gen, refs in
                  zip(generated_list, reference_list)]
        return sum(scores) / len(scores) if scores else None

    def compute_cider(self, generated_list, reference_list):
        return compute_cider_score(generated_list, reference_list) if generated_list else None

    # -------------------------------------------------------------------------
    # SCHEDULER STEP
    # -------------------------------------------------------------------------

    def _step_scheduler(self, epoch, val_loss, warmup_epochs):
        """
        Applique le step du scheduler selon le type.
        Retourne True si on est à la fin d'un cycle cosine (pour cosine uniquement).
        """
        _get_lr = lambda: self.optimizer.param_groups[0]['lr']
        _set_lr = lambda lr: [pg.update({'lr': lr})
                              for pg in self.optimizer.param_groups]

        lr_target = self.config['learning_rate']

        if epoch < warmup_epochs:
            new_lr = lr_target * (epoch + 1) / max(warmup_epochs, 1)
            _set_lr(new_lr)
            return False

        if self.scheduler_type == 'plateau':
            self.scheduler.step(val_loss)
            return False

        elif self.scheduler_type == 'cosine':
            # Corriger base_lrs une seule fois après le warmup
            if not self._cosine_initialized:
                self.scheduler.base_lrs = [lr_target] * len(self.scheduler.base_lrs)
                self._cosine_initialized = True

            cosine_epoch = epoch - warmup_epochs
            self.scheduler.step(cosine_epoch)

            # Détection de fin de cycle : T_cur revient à 0
            T0     = self.config.get('cosine_T0', 10)
            T_mult = self.config.get('cosine_T_mult', 2)
            t      = cosine_epoch + 1
            # Calculer les epochs de restart
            restart = T0
            while restart <= cosine_epoch + 1:
                if cosine_epoch + 1 == restart:
                    return True   # fin de cycle
                restart += T0 * (T_mult ** (restart // T0))
            # Méthode plus robuste : T_cur == 0 après step signifie restart
            if hasattr(self.scheduler, 'T_cur') and self.scheduler.T_cur == 0 and cosine_epoch > 0:
                return True

            return False

    # -------------------------------------------------------------------------
    # BOUCLE PRINCIPALE
    # -------------------------------------------------------------------------

    def train(self):
        print("\n" + "="*70)
        print(f"ENTRAÎNEMENT — model={self.config['model']}  scheduler={self.scheduler_type}")
        print("="*70)

        params = self.model.get_num_params()
        print(f"Paramètres : {params['total']:,}  "
              f"(encoder={params['encoder']:,}  decoder={params['decoder']:,})")
        print(f"Epochs     : {self.config['num_epochs']}")
        print(f"Batch size : {self.config['batch_size']}")
        print(f"LR         : {self.config['learning_rate']}")
        if self.scheduler_type == 'cosine':
            print(f"Cosine T0/T_mult : {self.config.get('cosine_T0',10)}/{self.config.get('cosine_T_mult',2)}")
            print(f"Max cycles sans amélioration : {self.config.get('max_no_improve_cycles', 3)}")
        else:
            print(f"Plateau patience : {self.config.get('plateau_patience', 10)}")
            print(f"Early stop patience : {self.config.get('patience', 15)}")
        print(f"Attention lambda : {self.config.get('attention_lambda', 1.0)}")
        print(f"Device     : {self.device}")

        bleu_every    = self.config.get('bleu_every', 1)
        warmup_epochs = self.config.get('warmup_epochs', 5)
        num_epochs    = self.config['num_epochs']

        start_time          = time.time()
        patience_counter    = 0  # pour plateau
        no_improve_cycles   = 0  # pour cosine
        last_cycle_best     = None  # meilleure métrique au début du cycle cosine courant

        _get_lr = lambda: self.optimizer.param_groups[0]['lr']

        # LR initial au début du warmup
        if warmup_epochs > 0:
            _set_lr = lambda lr: [pg.update({'lr': lr})
                                  for pg in self.optimizer.param_groups]
            _set_lr(self.config['learning_rate'] / max(warmup_epochs, 1))

        for epoch in range(num_epochs):

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            ppl = math.exp(min(val_loss, 20))
            self.perplexities.append(ppl)

            # ── Métriques ────────────────────────────────────────────────────
            bleu1 = bleu4 = meteor = cider = None

            if (epoch + 1) % bleu_every == 0:
                num_samples = self.config.get('bleu_num_samples', 5000)
                gen_list, ref_list = self._collect_predictions(num_samples)

                if gen_list:
                    bleu1, bleu4 = self.compute_bleu(gen_list, ref_list)
                    meteor = self.compute_meteor(gen_list, ref_list)
                    cider  = self.compute_cider(gen_list, ref_list)

                    ep = epoch + 1
                    if bleu1 is not None:
                        self.bleu1_scores.append((ep, bleu1))
                        self.bleu4_scores.append((ep, bleu4))
                        if bleu4 > self.best_bleu4:
                            self.best_bleu4 = bleu4
                    if meteor is not None:
                        self.meteor_scores.append((ep, meteor))
                        if meteor > self.best_meteor:
                            self.best_meteor = meteor
                    if cider is not None:
                        self.cider_scores.append((ep, cider))
                        if cider > self.best_cider:
                            self.best_cider = cider

            # Métrique de référence : val loss (plus bas = meilleur)
            # Les métriques BLEU/METEOR/CIDEr sont calculées et affichées
            # à titre informatif mais n'influencent plus l'early stopping
            # ni la sélection du meilleur modèle.

            # Suivi du meilleur dans le cycle cosine courant
            if self.scheduler_type == 'cosine' and epoch >= warmup_epochs:
                if val_loss < self._best_cycle_val_loss:
                    self._best_cycle_val_loss = val_loss

            # ── Scheduler step ───────────────────────────────────────────────
            end_of_cycle = self._step_scheduler(epoch, val_loss, warmup_epochs)
            lr_now = _get_lr()
            self.lr_history.append((epoch + 1, lr_now))

            # ── Affichage ────────────────────────────────────────────────────
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Val Loss   : {val_loss:.4f}  │  PPL : {ppl:.2f}"
                  + ("  ★ best" if val_loss <= self.best_val_loss else ""))
            print(f"  LR         : {lr_now:.2e}"
                  + ("  (warmup)" if epoch < warmup_epochs else ""))
            if bleu1  is not None:
                print(f"  BLEU-1     : {bleu1:.4f}")
                print(f"  BLEU-4     : {bleu4:.4f}")
            if meteor is not None:
                print(f"  METEOR     : {meteor:.4f}")
            if cider  is not None:
                print(f"  CIDEr      : {cider:.4f}")

            # ── Sauvegarde checkpoint périodique ─────────────────────────────
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                ckpt = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_model(self.model, ckpt, optimizer=self.optimizer,
                           epoch=epoch, loss=val_loss, vocab=self.vocabulary,
                           scheduler_state=self.scheduler.state_dict())

            # ── Meilleur modèle global (val loss) ────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(
                    self.model,
                    os.path.join(self.config['checkpoint_dir'], 'best_model.pth'),
                    optimizer=self.optimizer, epoch=epoch,
                    loss=val_loss, vocab=self.vocabulary,
                    scheduler_state=self.scheduler.state_dict()
                )
                print(f"  ✓ Nouveau meilleur modèle global (loss={val_loss:.4f})")

            # ── Meilleur modèle CIDEr (informatif, pas de référence) ─────────
            if cider is not None and cider >= self.best_cider:
                save_model(
                    self.model,
                    os.path.join(self.config['checkpoint_dir'], 'best_model_cider.pth'),
                    optimizer=self.optimizer, epoch=epoch,
                    loss=val_loss, vocab=self.vocabulary,
                    scheduler_state=self.scheduler.state_dict()
                )
                print(f"  ✓ Nouveau meilleur modèle CIDEr ({cider:.4f}) [informatif]")

            # ── Plots & historique ────────────────────────────────────────────
            if (epoch + 1) % bleu_every == 0:
                self.plot_learning_curves()
                self.save_history()

            # ── Early stopping ────────────────────────────────────────────────
            # Référence unique : val loss (plus bas = meilleur).
            improved = val_loss < self.best_val_loss + 1e-4

            if self.scheduler_type == 'plateau':
                # Early stop classique sur nb d'epochs sans amélioration val loss
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  ✗ Val loss sans amélioration ({patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config['patience']:
                    print("\nEarly stopping (plateau) déclenché !")
                    break

            elif self.scheduler_type == 'cosine' and end_of_cycle:
                # ── Fin de cycle : sauvegarder le meilleur modèle du cycle ────
                self._cycle_count += 1
                cycle_ckpt = os.path.join(
                    self.config['checkpoint_dir'],
                    f'best_model_cycle_{self._cycle_count}.pth'
                )
                # On sauvegarde le best_model.pth courant sous le nom du cycle
                # (best_model.pth contient déjà le meilleur de tout l'entraînement,
                #  donc le meilleur de ce cycle si la val loss a baissé dans ce cycle)
                import shutil
                best_global = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                if os.path.exists(best_global):
                    shutil.copy2(best_global, cycle_ckpt)
                    self._cycle_checkpoints.append(cycle_ckpt)
                    print(f"  → Checkpoint fin de cycle {self._cycle_count} sauvegardé : {cycle_ckpt}")

                # Réinitialiser le suivi du cycle
                self._best_cycle_val_loss = float('inf')

                # ── Early stop sur nb de cycles sans amélioration val loss ────
                cycle_val = self.best_val_loss  # meilleure val loss globale à ce point
                if last_cycle_best is None or cycle_val < last_cycle_best - 1e-4:
                    last_cycle_best   = cycle_val
                    no_improve_cycles = 0
                    print(f"  ✓ Fin de cycle cosine — amélioration val loss détectée")
                else:
                    no_improve_cycles += 1
                    max_c = self.config.get('max_no_improve_cycles', 3)
                    print(f"  ✗ Fin de cycle cosine — val loss stagnante "
                          f"({no_improve_cycles}/{max_c} cycles)")
                    if no_improve_cycles >= max_c:
                        print(f"\nEarly stopping (cosine) déclenché après "
                              f"{no_improve_cycles} cycles sans amélioration !")
                        break

        # ── Résumé final ──────────────────────────────────────────────────────
        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/60:.2f} minutes")
        print(f"Meilleure val loss  : {self.best_val_loss:.4f}  │  PPL : {math.exp(min(self.best_val_loss,20)):.2f}")
        if self.best_bleu4  > 0: print(f"Meilleur BLEU-4    : {self.best_bleu4:.4f}  [informatif]")
        if self.best_meteor > 0: print(f"Meilleur METEOR    : {self.best_meteor:.4f}  [informatif]")
        if self.best_cider  > 0: print(f"Meilleur CIDEr     : {self.best_cider:.4f}  [informatif]")

        # ── Moyenne des checkpoints de cycle (cosine uniquement) ─────────────
        if self.scheduler_type == 'cosine' and len(self._cycle_checkpoints) >= 2:
            self._average_cycle_checkpoints()

        self.plot_learning_curves()
        self.save_history()

    # -------------------------------------------------------------------------
    # MOYENNE DES CHECKPOINTS DE CYCLE (SWA simplifié)
    # -------------------------------------------------------------------------

    def _average_cycle_checkpoints(self):
        """
        Moyenne les poids de tous les checkpoints sauvegardés en fin de cycle cosine.

        Principe (Stochastic Weight Averaging, Izmailov et al. 2018) :
          Chaque cycle cosine converge vers un minimum légèrement différent.
          La moyenne arithmétique des poids de ces minima tombe typiquement
          dans une zone plus plate de la loss surface → meilleure généralisation
          qu'un seul minimum.

        Le modèle moyenné est sauvegardé dans averaged_model.pth.
        Sa val loss est évaluée et affichée pour comparaison.

        Ne s'exécute que si au moins 2 cycles ont été complétés.
        """
        n = len(self._cycle_checkpoints)
        print(f"\n{'='*70}")
        print(f"MOYENNE DES POIDS ({n} checkpoints de cycle)")
        print(f"{'='*70}")
        for p in self._cycle_checkpoints:
            print(f"  · {os.path.basename(p)}")

        # Charger les state_dicts de tous les cycles
        state_dicts = []
        vocab_from_ckpt = None
        for ckpt_path in self._cycle_checkpoints:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                state_dicts.append({
                    'encoder': ckpt['encoder_state_dict'],
                    'decoder': ckpt['decoder_state_dict'],
                })
                if vocab_from_ckpt is None:
                    vocab_from_ckpt = ckpt.get('vocab')
            except Exception as e:
                print(f"  ⚠️  Impossible de charger {ckpt_path} : {e}")

        if len(state_dicts) < 2:
            print("  Pas assez de checkpoints valides — moyenne ignorée.")
            return

        # Moyenne arithmétique paramètre par paramètre
        print(f"\nCalcul de la moyenne sur {len(state_dicts)} checkpoints...")

        def average_state_dicts(dicts):
            avg = {}
            for key in dicts[0]:
                tensors = [d[key].float() for d in dicts]
                avg[key] = torch.stack(tensors, dim=0).mean(dim=0)
                # Reconvertir dans le dtype d'origine
                avg[key] = avg[key].to(dicts[0][key].dtype)
            return avg

        avg_enc = average_state_dicts([sd['encoder'] for sd in state_dicts])
        avg_dec = average_state_dicts([sd['decoder'] for sd in state_dicts])

        # Charger dans le modèle courant
        self.model.encoder.load_state_dict(avg_enc)
        self.model.decoder.load_state_dict(avg_dec)
        self.model.to(self.device)

        # Évaluer la val loss du modèle moyenné
        print("Évaluation du modèle moyenné...")
        avg_val_loss = self.validate()
        avg_ppl      = math.exp(min(avg_val_loss, 20))

        print(f"\n  Val Loss modèle moyenné : {avg_val_loss:.4f}  │  PPL : {avg_ppl:.2f}")
        print(f"  Val Loss meilleur cycle : {self.best_val_loss:.4f}  │  PPL : {math.exp(min(self.best_val_loss,20)):.2f}")

        if avg_val_loss < self.best_val_loss:
            print(f"  → Le modèle moyenné est MEILLEUR (+{self.best_val_loss - avg_val_loss:.4f})")
        else:
            delta = avg_val_loss - self.best_val_loss
            print(f"  → Le modèle moyenné est légèrement moins bon (+{delta:.4f})")
            print(f"     (normal si les cycles ont convergé vers des minima très différents)")

        # Sauvegarder le modèle moyenné
        avg_path = os.path.join(self.config['checkpoint_dir'], 'averaged_model.pth')
        from models.caption_model import save_model as _save
        _save(
            self.model, avg_path,
            epoch=None, loss=avg_val_loss,
            vocab=vocab_from_ckpt or self.vocabulary,
        )
        print(f"  → Modèle moyenné sauvegardé : {avg_path}")
        print(f"{'='*70}")

    # -------------------------------------------------------------------------
    # PLOTS
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
            f'Entraînement COCO — {self.config["model"]} / {self.scheduler_type}',
            fontsize=14, fontweight='bold', y=1.02
        )

        panel = 0

        ax = axes[panel]; panel += 1
        ax.plot(epochs_all, self.train_losses, 'b-', label='Train', linewidth=2)
        ax.plot(epochs_all, self.val_losses,   'r-', label='Val',   linewidth=2)
        if self.val_losses:
            best_ep = self.val_losses.index(min(self.val_losses)) + 1
            ax.axvline(x=best_ep, color='green', linestyle='--', alpha=0.5,
                       label=f'Best (ep.{best_ep})')
        ax.set_title('Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[panel]; panel += 1
        ax.plot(epochs_all, self.perplexities, color='purple', linewidth=2)
        ax.set_title('Perplexité', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('PPL')
        ax.grid(True, alpha=0.3)

        ax = axes[panel]; panel += 1
        if self.lr_history:
            lr_epochs = [e for e, _ in self.lr_history]
            lr_vals   = [lr for _, lr in self.lr_history]
            ax.plot(lr_epochs, lr_vals, color='teal', linewidth=2)
            ax.set_yscale('log')
            _wu = self.config.get('warmup_epochs', 5)
            if _wu > 0 and lr_epochs:
                ax.axvspan(1, min(_wu, max(lr_epochs)),
                           alpha=0.10, color='orange', label=f'Warmup ({_wu}ep.)')
            if self.scheduler_type == 'cosine':
                T0     = self.config.get('cosine_T0', 10)
                T_mult = self.config.get('cosine_T_mult', 2)
                t = T0
                while _wu + t <= (max(lr_epochs) if lr_epochs else 1):
                    ax.axvline(x=_wu + t, color='steelblue', linestyle=':',
                               alpha=0.6, linewidth=1.2,
                               label='Cosine restart' if t == T0 else '')
                    t *= T_mult
        ax.set_title('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('LR (log)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

        if has_bleu:
            ax = axes[panel]; panel += 1
            bep = [e for e, _ in self.bleu1_scores]
            ax.plot(bep, [s for _, s in self.bleu1_scores], 'b-o', label='BLEU-1', linewidth=2, markersize=5)
            ax.plot(bep, [s for _, s in self.bleu4_scores], 'r-o', label='BLEU-4', linewidth=2, markersize=5)
            ax.set_title('BLEU', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

        if has_meteor:
            ax = axes[panel]; panel += 1
            mep = [e for e, _ in self.meteor_scores]
            ax.plot(mep, [s for _, s in self.meteor_scores], 'g-o', linewidth=2, markersize=5)
            ax.set_title('METEOR', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)

        if has_cider:
            ax = axes[panel]; panel += 1
            cep = [e for e, _ in self.cider_scores]
            ax.plot(cep, [s for _, s in self.cider_scores], color='darkorange',
                    marker='o', linewidth=2, markersize=5)
            ax.set_title('CIDEr', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(
            self.config['log_dir'],
            f'curves_{self.config["model"]}_{self.scheduler_type}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_history(self):
        history = {
            'model':         self.config['model'],
            'scheduler':     self.scheduler_type,
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
            'config':        {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in self.config.items()},
        }
        path = os.path.join(
            self.config['log_dir'],
            f'history_{self.config["model"]}_{self.scheduler_type}.json'
        )
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Entraîne un modèle d\'image captioning sur COCO.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python train.py --model densenet --scheduler cosine
  python train.py --model resnet   --scheduler plateau
  python train.py --model cnn      --scheduler cosine  --fast
  python train.py --model densenet --scheduler cosine  --resume checkpoints/densenet/best_model.pth
        """
    )
    parser.add_argument('--model',     choices=['cnn', 'resnet', 'densenet'],
                        default='densenet',
                        help='Architecture du modèle (défaut: densenet)')
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'],
                        default='cosine',
                        help='Scheduler LR (défaut: cosine)')
    parser.add_argument('--fast',      action='store_true',
                        help='Mode développement rapide (peu d\'epochs, petit vocab)')
    parser.add_argument('--resume',    type=str, default=None,
                        help='Chemin vers un checkpoint pour reprendre l\'entraînement')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.model, fast=args.fast)

    # Surcharger le checkpoint_dir/log_dir avec le nom du scheduler
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], args.scheduler)
    config['log_dir']        = os.path.join(config['log_dir'], args.scheduler)

    print("="*70)
    print(f"IMAGE CAPTIONING COCO")
    print(f"  Modèle    : {args.model}")
    print(f"  Scheduler : {args.scheduler}")
    print(f"  Fast mode : {args.fast}")
    print("="*70)

    # ── Vocabulaire ───────────────────────────────────────────────────────────
    if os.path.exists(config['vocab_path']):
        print(f"\nChargement du vocabulaire depuis {config['vocab_path']}")
        vocab = Vocabulary.load(config['vocab_path'])
    else:
        print("\nConstruction du vocabulaire (train set COCO)...")
        train_cap_prep = CaptionPreprocessor(
            config['train_captions_file'], config['train_images_dir']
        )
        vocab = Vocabulary(freq_threshold=config['freq_threshold'])
        vocab.build_vocabulary(train_cap_prep.get_all_captions())
        vocab.save(config['vocab_path'])

    print(f"Vocabulaire : {len(vocab)} mots")

    # ── Données ───────────────────────────────────────────────────────────────
    print("\nChargement des paires train...")
    train_cap_prep = CaptionPreprocessor(
        config['train_captions_file'], config['train_images_dir']
    )
    train_pairs = train_cap_prep.get_image_caption_pairs()

    print("Chargement des paires val...")
    val_cap_prep = CaptionPreprocessor(
        config['val_captions_file'], config['val_images_dir']
    )
    val_pairs = val_cap_prep.get_image_caption_pairs()

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_prep = ImagePreprocessor(
        image_size=config['image_size'], normalize=False,
        train_transform=train_transform, val_transform=val_transform,
    )

    train_loader, val_loader = get_data_loaders(
        train_pairs=train_pairs, val_pairs=val_pairs,
        vocabulary=vocab, image_preprocessor=image_prep,
        batch_size=config['batch_size'], num_workers=config['num_workers'],
        shuffle_train=True
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    print(f"\nCréation du modèle ({args.model})...")
    model = create_model(
        vocab_size    = len(vocab),
        embedding_dim = config['embedding_dim'],
        hidden_dim    = config['hidden_dim'],
        feature_dim   = config['feature_dim'],
        dropout       = config['dropout'],
        model         = config['model'],
        attention_dim = config.get('attention_dim', 256),
        growth_rate   = config.get('growth_rate',   32),
        compression   = config.get('compression',   0.5),
        dense_dropout = config.get('dense_dropout', 0.0),
        block_config  = config.get('block_config',  (6, 12, 24, 16)),
    )

    # ── Reprise d'entraînement ────────────────────────────────────────────────
    if args.resume:
        print(f"\nReprise depuis {args.resume}...")
        _, info = load_model(args.resume, device='cpu')
        model.load_state_dict(
            torch.load(args.resume, map_location='cpu', weights_only=False)['encoder_state_dict'],
            strict=False
        )
        print(f"  Epoch précédente : {info.get('epoch', '?')}")

    # ── Entraînement ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        vocabulary=vocab, config=config,
        scheduler_type=args.scheduler,
        val_pairs=val_pairs
    )
    trainer.train()

    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ !")
    print("="*70)


if __name__ == "__main__":
    main()