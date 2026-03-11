"""
Script d'entraînement COCO pour Image Captioning
=================================================

Compatible avec les quatre encoder_type :
  'lite'      → EncoderCNNLite   + DecoderLSTM
  'full'      → EncoderCNN       + DecoderLSTM
  'attention' → EncoderSpatial   + DecoderWithAttention
  'densenet'  → EncoderDenseNet  + DecoderWithAttention   ← recommandé

Correctifs v4 (DenseNet) :
  - La régularisation doubly stochastic s'active pour encoder_type='attention'
    ET encoder_type='densenet' (les deux utilisent DecoderWithAttention).
  - Les paramètres DenseNet (growth_rate, compression, dense_dropout,
    block_config) sont passés à create_model() depuis CONFIG.
  - Le reste du Trainer est inchangé.
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

CIDER_AVAILABLE = True  # implémentation interne

from utils import vocabulary, data_loader
from utils.preprocessing_coco import CaptionPreprocessor, ImagePreprocessor
from models2 import caption_model2

from config_coco3 import CONFIG


# =============================================================================
# IMPLÉMENTATION CIDER LÉGÈRE
# =============================================================================

def _ngrams(words, n):
    from collections import Counter
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def compute_cider_score(generated_list, reference_list, n_max=4):
    """
    Calcule le score CIDEr-D sur un corpus.
    Utilise TF-IDF pour pondérer les n-grammes rares (plus informatifs).
    """
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
            label_smoothing=0.1
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )

        # CosineAnnealingWarmRestarts — remplace ReduceLROnPlateau(patience=2)
        # T_0=10 : premier cycle de 10 epochs après le warmup
        # T_mult=2 : les cycles doublent → 10, 20, 40...
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('cosine_T0', 10),
            T_mult=config.get('cosine_T_mult', 2),
            eta_min=config.get('lr_min', 1e-5),
        )

        # La régularisation doubly stochastic s'active pour les deux modes
        # qui utilisent DecoderWithAttention : 'attention' et 'densenet'.
        _etype = config.get('encoder_type', '')
        self._use_attention_reg = (
            _etype in ('attention', 'densenet')
            and config.get('attention_lambda', 1.0) > 0.0
        )
        self._attention_lambda = config.get('attention_lambda', 1.0)

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

            inputs  = captions[:, :-1]
            targets = captions[:, 1:]

            if self._use_attention_reg:
                outputs, alphas = self.model.forward_with_alphas(images, inputs)
            else:
                outputs = self.model(images, inputs)
                alphas  = None

            outputs_flat = outputs.reshape(-1, outputs.shape[2])
            targets_flat = targets.reshape(-1)

            loss = self.criterion(outputs_flat, targets_flat)

            # Régularisation doubly stochastic (Xu et al. 2015)
            # Pénalise l'attention qui se concentre sur les mêmes régions
            # à chaque pas au lieu de couvrir uniformément la grille.
            if alphas is not None:
                attention_sum = alphas.sum(dim=1)              # (B, P)
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

                outputs = self.model(images, inputs)
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / num_batches

    # -------------------------------------------------------------------------

    def _collect_predictions(self, num_samples):
        """
        Génère des captions pour num_samples images de la validation.
        Regroupe toutes les références COCO (5/image) pour un CIDEr fiable.
        """
        self.model.eval()

        start_token = self.vocabulary.word2idx[self.vocabulary.start_token]
        end_token   = self.vocabulary.word2idx[self.vocabulary.end_token]
        pad_token   = self.vocabulary.word2idx[self.vocabulary.pad_token]

        image_refs  = {}
        image_order = []

        global_idx = 0
        for images, captions, lengths in self.val_loader:
            for i in range(images.size(0)):
                ref_ids   = [t.item() for t in captions[i]
                             if t.item() not in [start_token, end_token, pad_token]]
                ref_words = self.vocabulary.denumericalize(ref_ids).split()
                if not ref_words:
                    global_idx += 1
                    continue

                if global_idx not in image_refs:
                    image_refs[global_idx] = []
                    image_order.append(global_idx)
                image_refs[global_idx].append(ref_words)
                global_idx += 1

            if len(image_order) >= num_samples:
                break

        image_order = image_order[:num_samples]

        generated_list = []
        reference_list = []
        global_idx = 0
        ptr = 0

        with torch.no_grad():
            for images, captions, lengths in self.val_loader:
                if ptr >= len(image_order):
                    break
                images_gpu = images.to(self.device)

                for i in range(images.size(0)):
                    if ptr >= len(image_order):
                        break
                    if global_idx != image_order[ptr]:
                        global_idx += 1
                        continue

                    features  = self.model.encoder(images_gpu[i:i+1])
                    generated = self.model.decoder.generate(
                        features,
                        max_length=self.config.get('max_caption_length', 20),
                        start_token=start_token,
                        end_token=end_token
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
                        reference_list.append(image_refs[global_idx])

                    ptr        += 1
                    global_idx += 1

        return generated_list, reference_list

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

        etype = self.config['encoder_type']
        print(f"\nConfiguration :")
        print(f"  Encoder type        : {etype}")
        if etype == 'densenet':
            print(f"  DenseNet block_cfg  : {self.config.get('block_config')}")
            print(f"  growth_rate / θ     : {self.config.get('growth_rate')} / {self.config.get('compression')}")
        print(f"  Epochs              : {self.config['num_epochs']}")
        print(f"  Batch size          : {self.config['batch_size']}")
        print(f"  Learning rate       : {self.config['learning_rate']}")
        print(f"  Cosine T0/T_mult    : {self.config.get('cosine_T0',10)}/{self.config.get('cosine_T_mult',2)}")
        print(f"  Attention lambda    : {self.config.get('attention_lambda', 1.0)}")
        print(f"  bleu_num_samples    : {self.config.get('bleu_num_samples', 5000)}")
        print(f"  Device              : {self.device}")

        bleu_every    = self.config.get('bleu_every', 2)
        warmup_epochs = self.config.get('warmup_epochs', 5)

        start_time       = time.time()
        patience_counter = 0

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

            lr_before = _get_lr()

            if epoch < warmup_epochs:
                new_lr = _lr_target * (epoch + 1) / warmup_epochs
                _set_lr(new_lr)
                lr_note = f"warmup {epoch+1}/{warmup_epochs}"
            else:
                cosine_epoch = epoch - warmup_epochs
                self.scheduler.step(cosine_epoch)
                lr_note = "CosineAnnealingWarmRestarts"

            lr_after   = _get_lr()
            lr_changed = lr_after < lr_before * 0.99
            self.lr_history.append((epoch + 1, lr_after))

            ppl = math.exp(val_loss)
            self.perplexities.append(ppl)

            bleu1 = bleu4 = meteor = cider = None

            if (epoch + 1) % bleu_every == 0:
                num_samples = self.config.get('bleu_num_samples', 5000)
                generated_list, reference_list = self._collect_predictions(num_samples)

                if generated_list:
                    bleu1, bleu4 = self.compute_bleu(generated_list, reference_list)
                    meteor = self.compute_meteor(generated_list, reference_list)
                    cider  = self.compute_cider(generated_list, reference_list)

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

            if (epoch + 1) % bleu_every == 0:
                self.plot_learning_curves()
                self.save_history()

            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                ckpt_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                caption_model2.save_model(
                    self.model, ckpt_path, optimizer=self.optimizer,
                    epoch=epoch, loss=val_loss, vocab=self.vocabulary
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                caption_model2.save_model(
                    self.model,
                    os.path.join(self.config['checkpoint_dir'], 'best_model.pth'),
                    optimizer=self.optimizer, epoch=epoch,
                    loss=val_loss, vocab=self.vocabulary
                )
                print(f"  ✓ Nouveau meilleur modèle (loss : {val_loss:.4f})")

            if cider is not None and cider >= self.best_cider:
                caption_model2.save_model(
                    self.model,
                    os.path.join(self.config['checkpoint_dir'], 'best_model_cider.pth'),
                    optimizer=self.optimizer, epoch=epoch,
                    loss=val_loss, vocab=self.vocabulary
                )
                print(f"  ✓ Nouveau meilleur modèle CIDEr ({cider:.4f})")

            # Early stopping — priorité CIDEr si disponible, sinon val loss
            if cider is not None:
                if cider >= self.best_cider:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  ✗ CIDEr sans amélioration "
                          f"(patience : {patience_counter}/{self.config['patience']})")
            else:
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

        ax = axes[panel]; panel += 1
        ax.plot(epochs_all, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs_all, self.val_losses,   'r-', label='Val Loss',   linewidth=2)
        best_epoch = self.val_losses.index(min(self.val_losses)) + 1
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                   label=f'Best (ep.{best_epoch})')
        ax.set_title('Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy Loss')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[panel]; panel += 1
        ax.plot(epochs_all, self.perplexities, color='purple', linewidth=2)
        ax.set_title('Perplexité (exp(val_loss))', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('PPL')
        ax.grid(True, alpha=0.3)
        best_ppl_idx = self.perplexities.index(min(self.perplexities))
        ax.annotate(f'PPL={min(self.perplexities):.1f} (best)',
                    xy=(best_ppl_idx + 1, min(self.perplexities)),
                    fontsize=8, color='green', xytext=(5, 10),
                    textcoords='offset points')

        ax = axes[panel]; panel += 1
        if self.lr_history:
            lr_epochs = [e for e, _ in self.lr_history]
            lr_vals   = [lr for _, lr in self.lr_history]
            ax.plot(lr_epochs, lr_vals, color='teal', linewidth=2)
            ax.set_yscale('log')
            _wu = self.config.get('warmup_epochs', 5)
            if _wu > 0 and max(lr_epochs) >= 1:
                ax.axvspan(1, min(_wu, max(lr_epochs)),
                           alpha=0.10, color='orange', label=f'Warmup ({_wu} ep.)')
            T0     = self.config.get('cosine_T0', 10)
            T_mult = self.config.get('cosine_T_mult', 2)
            t = T0
            restart_ep = _wu + t
            while restart_ep <= max(lr_epochs):
                ax.axvline(x=restart_ep, color='steelblue', linestyle=':',
                           alpha=0.6, linewidth=1.2,
                           label='Cosine restart' if t == T0 else '')
                t *= T_mult
                restart_ep = _wu + t
        ax.set_title('Learning Rate (Cosine + Warmup)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('LR (log scale)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

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
                        fontsize=8, color='red', xytext=(8, -20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))

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
                        fontsize=8, color='green', xytext=(8, -20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green', lw=1))

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
                        fontsize=8, color='darkorange', xytext=(8, -20),
                        textcoords='offset points',
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
            'config':        {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in self.config.items()},
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

    if os.path.exists(CONFIG['vocab_path']):
        print(f"\nChargement du vocabulaire depuis {CONFIG['vocab_path']}")
        vocab = vocabulary.Vocabulary.load(CONFIG['vocab_path'])
    else:
        print("\nConstruction du vocabulaire depuis le train set COCO...")
        train_caption_prep = CaptionPreprocessor(
            CONFIG['train_captions_file'], CONFIG['train_images_dir']
        )
        vocab = vocabulary.Vocabulary(freq_threshold=CONFIG['freq_threshold'])
        vocab.build_vocabulary(train_caption_prep.get_all_captions())
        vocab.save(CONFIG['vocab_path'])

    print(f"Taille du vocabulaire : {len(vocab)}")

    print("\nChargement des paires train (COCO train2017)...")
    train_caption_prep = CaptionPreprocessor(
        CONFIG['train_captions_file'], CONFIG['train_images_dir']
    )
    train_pairs = train_caption_prep.get_image_caption_pairs()

    print("\nChargement des paires val (COCO val2017)...")
    val_caption_prep = CaptionPreprocessor(
        CONFIG['val_captions_file'], CONFIG['val_images_dir']
    )
    val_pairs = val_caption_prep.get_image_caption_pairs()

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
        image_size=CONFIG['image_size'], normalize=False,
        train_transform=train_transform, val_transform=val_transform,
    )

    train_loader, val_loader = data_loader.get_data_loaders(
        train_pairs=train_pairs, val_pairs=val_pairs,
        vocabulary=vocab, image_preprocessor=image_prep,
        batch_size=CONFIG['batch_size'], num_workers=CONFIG['num_workers'],
        shuffle_train=True
    )

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
        # Paramètres DenseNet (ignorés si encoder_type != 'densenet')
        growth_rate   = CONFIG.get('growth_rate',   32),
        compression   = CONFIG.get('compression',   0.5),
        dense_dropout = CONFIG.get('dense_dropout', 0.0),
        block_config  = CONFIG.get('block_config',  (6, 12, 24, 16)),
    )

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        vocabulary=vocab, config=CONFIG
    )

    trainer.train()

    print("\n" + "="*70)
    print("ENTRAÎNEMENT COCO TERMINÉ !")
    print("="*70)


if __name__ == "__main__":
    main()