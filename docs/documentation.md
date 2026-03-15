# Image Captioning COCO — Documentation

> **Dataset** : COCO 2017 · **Encodeurs** : CNN · ResNet · DenseNet · **Décodeur** : LSTM + Bahdanau · **Schedulers** : Cosine · Plateau

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure du projet](#2-structure-du-projet)
3. [Vocabulaire et données](#3-vocabulaire-et-données)
4. [Encodeurs CNN](#4-encodeurs-cnn)
5. [Décodeurs LSTM](#5-décodeurs-lstm)
6. [Schedulers LR](#6-schedulers-lr)
7. [Boucle d'entraînement](#7-boucle-dentraînement)
8. [Inférence et évaluation](#8-inférence-et-évaluation)
9. [Commandes CLI](#9-commandes-cli)
10. [Hyperparamètres de référence](#10-hyperparamètres-de-référence)

---

## 1. Vue d'ensemble

### Pipeline global

```
Image (224×224×3)
        │
        ▼
┌───────────────┐
│   ENCODEUR    │   CNN / ResNet / DenseNet (from scratch)
│  feature_dim  │
└───────┬───────┘
        │ (B,512) ou (B,49,512)
        ▼
┌───────────────┐
│   DÉCODEUR    │   LSTM + Attention de Bahdanau
│   LSTM Cell   │
└───────┬───────┘
        │
        ▼
  Caption texte   "a dog running in the park"
```

### Les trois architectures

| `--model` | Encodeur | Décodeur | Sortie encodeur | Attention | Params |
|-----------|----------|----------|-----------------|-----------|--------|
| `cnn` | EncoderCNN (résiduel) | DecoderLSTM | `(B, 512)` — vecteur global | ✗ | ~7 M |
| `resnet` | EncoderSpatial (résiduel) | DecoderWithAttention | `(B, 49, 512)` — grille 7×7 | ✓ Bahdanau | ~10 M |
| `densenet` ★ | EncoderDenseNet (DenseNet-121) | DecoderWithAttention | `(B, 49, 512)` — grille 7×7 | ✓ Bahdanau | ~18 M |

> **Tous les encodeurs sont entraînés from scratch** — aucun poids pré-entraîné (ImageNet ou autre).

### Les deux schedulers

| `--scheduler` | Algorithme | Early stopping | Spécificité |
|---------------|------------|----------------|-------------|
| `cosine` | CosineAnnealingWarmRestarts | Après N cycles stagnants | SWA — moyenne des poids de cycle |
| `plateau` | ReduceLROnPlateau | Après N epochs stagnantes | Réduction LR × 0.5 sur plateau |

> **Métrique de référence unique : val loss.** BLEU-1/4, METEOR et CIDEr sont calculés à titre informatif mais n'influencent ni l'early stopping ni les checkpoints.

---

## 2. Structure du projet

```
image_captioning/
│
├── config.py                  ← Configurations unifiées (3 archi × 2 schedulers)
├── train.py                   ← Trainer, schedulers, SWA
├── demo.py                    ← Génération de captions sur images
├── evaluate.py                ← BLEU / METEOR / CIDEr sur val2017
├── visualize_attention.py     ← Cartes d'attention Bahdanau
├── prepare_data.py            ← Construction vocabulaire (1 fois)
├── test.py                    ← 70 tests unitaires (11 groupes)
│
├── models/
│   ├── encoder.py             ← EncoderCNN · EncoderSpatial · EncoderDenseNet
│   ├── decoder.py             ← DecoderLSTM · DecoderWithAttention (Bahdanau)
│   └── caption_model.py       ← create_model · save_model · load_model
│
├── utils/
│   ├── vocabulary.py          ← Construction vocab · numericalize · save/load
│   ├── preprocessing.py       ← ImagePreprocessor · CaptionPreprocessor (JSON COCO)
│   └── data_loader.py         ← ImageCaptionDataset · CaptionCollate · get_data_loaders
│
└── data/
    ├── coco_vocab.pkl         ← généré par prepare_data.py
    └── coco/
        ├── annotations/
        │   ├── captions_train2017.json
        │   └── captions_val2017.json
        ├── train2017/         ← ~118 000 images
        └── val2017/           ← ~5 000 images
```

### Checkpoints générés (cosine)

```
checkpoints/densenet/cosine/
├── best_model.pth             ← meilleure val loss (référence)
├── best_model_cider.pth       ← meilleur CIDEr [informatif]
├── best_model_cycle_1.pth     ← meilleur modèle du cycle 1
├── best_model_cycle_2.pth     ← meilleur modèle du cycle 2
├── averaged_model.pth         ← SWA : moyenne des cycles
└── checkpoint_epoch_N.pth     ← périodique (save_every=5)
```

---

## 3. Vocabulaire et données

### Tokens spéciaux (indices fixes)

| Token | Index | Rôle |
|-------|-------|------|
| `<PAD>` | 0 | Padding pour égaliser les longueurs dans un batch |
| `<START>` | 1 | Début de séquence |
| `<END>` | 2 | Fin de séquence |
| `<UNK>` | 3 | Mot inconnu (en dessous du seuil de fréquence) |

### Construction du vocabulaire

```python
from utils.vocabulary import Vocabulary

vocab = Vocabulary(freq_threshold=5)   # mots apparaissant < 5 fois → <UNK>
vocab.build_vocabulary(all_captions)   # ~10 000 mots sur COCO
vocab.save("data/coco_vocab.pkl")

# Conversion texte ↔ indices
indices = vocab.numericalize("a dog running")
# → [1, 45, 123, 89, 2]   (START, a, dog, running, END)

text = vocab.denumericalize(indices)
# → "a dog running"
```

### Format COCO (CaptionPreprocessor)

Le fichier d'annotations COCO est un JSON structuré :

```json
{
  "images": [{"id": 391895, "file_name": "000000391895.jpg"}, ...],
  "annotations": [{"image_id": 391895, "caption": "a man riding..."}, ...]
}
```

Chaque image a **5 captions humaines** — toutes utilisées pour le calcul du CIDEr.

### DataLoader et padding

Les captions ont des longueurs variables. `CaptionCollate` les aligne par padding :

```
Caption 1 : [1, 45, 123, 2]            longueur = 4
Caption 2 : [1, 34, 56, 78, 90, 2]     longueur = 6

Après padding (max_len=6) :
Caption 1 : [1, 45, 123, 2, 0, 0]
Caption 2 : [1, 34, 56, 78, 90, 2]

Batch retourné : (images, captions_padded, lengths)
  images   : (B, 3, 224, 224)
  captions : (B, max_seq_len)
  lengths  : (B,)             ← longueurs réelles avant padding
```

---

## 4. Encodeurs CNN

### 4.1 EncoderCNN — `model='cnn'`

Sortie vectorielle globale `(B, feature_dim)`. Utilisé avec `DecoderLSTM`.

```
Input : (B, 3, 224, 224)
  ↓ Conv7×7 stride2 + BN + ReLU + MaxPool   → (B, 64, 56, 56)
  ↓ ResidualBlock(64 → 128, stride=2)        → (B, 128, 28, 28)
  ↓ ResidualBlock(128 → 256, stride=2)       → (B, 256, 14, 14)
  ↓ ResidualBlock(256 → 512, stride=2)       → (B, 512,  7,  7)
  ↓ ResidualBlock(512 → 512, stride=1)       → (B, 512,  7,  7)
  ↓ AdaptiveAvgPool(1,1) + Flatten           → (B, 512)
  ↓ FC(512→feature_dim) + ReLU + Dropout
Output : (B, 512)
```

### 4.2 EncoderSpatial — `model='resnet'`

Même architecture que EncoderCNN, mais retourne une **grille spatiale 7×7** au lieu d'un vecteur global. Compatible avec `DecoderWithAttention`.

```
  ...  (identique à EncoderCNN jusqu'au dernier ResidualBlock)
  ↓ AdaptiveAvgPool(7,7)                     → (B, 512, 7, 7)
  ↓ reshape + FC pixelwise(512→feature_dim)
Output : (B, 49, 512)   ← 49 régions spatiales
```

### 4.3 EncoderDenseNet — `model='densenet'` ★

Architecture **DenseNet-121** (Huang et al. 2017). Chaque couche reçoit la concaténation de toutes les sorties des couches précédentes du même bloc.

```
Input : (B, 3, 224, 224)
  ↓ Stem : Conv7×7/2 + BN + ReLU + MaxPool3×3/2    → (B, 64, 56, 56)
  ↓ DenseBlock1 :  6 layers, growth_rate=32         → (B, 256, 56, 56)
  ↓ Transition1 : Conv1×1 (θ=0.5) + AvgPool/2      → (B, 128, 28, 28)
  ↓ DenseBlock2 : 12 layers                         → (B, 512, 28, 28)
  ↓ Transition2                                     → (B, 256, 14, 14)
  ↓ DenseBlock3 : 24 layers                         → (B, 1024, 14, 14)
  ↓ Transition3                                     → (B, 512,  7,  7)
  ↓ DenseBlock4 : 16 layers                         → (B, 1024,  7,  7)
  ↓ BN + ReLU + AdaptiveAvgPool(7,7)
  ↓ FC pixelwise(1024→feature_dim)
Output : (B, 49, 512)
```

**Connexion dense dans un DenseBlock :**

```python
# Chaque DenseLayer reçoit toutes les sorties précédentes
x_2 = cat([x_0, x_1, DenseLayer(cat([x_0, x_1]))])
x_3 = cat([x_0, x_1, x_2, DenseLayer(cat([x_0, x_1, x_2]))])
# ...
```

> **Avantage pour le captioning :** la grille 7×7 finale agrège des features à toutes les échelles. Pour générer "airplane", le décodeur peut s'appuyer sur des features de forme haut-niveau ; pour "runway", sur des features de texture bas-niveau — dans la même passe forward.

### Bloc résiduel

```
ResidualBlock(in_ch, out_ch, stride) :
  out = Conv3×3(BN(x)) → BN → ReLU → Conv3×3 → BN
  shortcut = Identity()              si in_ch == out_ch et stride == 1
           = Conv1×1 + BN            sinon
  return ReLU(out + shortcut(x))
```

---

## 5. Décodeurs LSTM

### 5.1 DecoderLSTM — `model='cnn'`

Decoder LSTM multi-couches classique. L'état caché initial est projeté depuis le vecteur de features global.

```python
# Initialisation
h0 = tanh(feature_projection(features))    # (num_layers, B, hidden_dim)
c0 = zeros_like(h0)

# Teacher forcing (entraînement)
embeddings = Dropout(Embedding(captions))  # (B, T, emb_dim)
lstm_out, _ = LSTM(embeddings, (h0, c0))   # (B, T, hidden_dim)
logits = FC(Dropout(lstm_out))             # (B, T, vocab_size)
```

Supports : **greedy search** et **beam search** (`generate_beam_search`).

### 5.2 DecoderWithAttention — `model='resnet'` ou `'densenet'`

Utilise un **LSTMCell** (1 seule couche) et recalcule le contexte visuel à chaque pas de temps via l'**attention de Bahdanau**.

#### Initialisation

```python
mean_feat = features.mean(dim=1)          # (B, feature_dim)
h = tanh(init_h(mean_feat))               # (B, hidden_dim)
c = tanh(init_c(mean_feat))
```

#### Pas de temps t

```
1. context, alpha = BahdanauAttention(features, h)
   ├── e_i   = v · tanh(W_enc·f_i + W_dec·h)     pour i=1..49
   ├── alpha = softmax(e)                           (B, 49)
   └── ctx   = Σ alpha_i · f_i                     (B, feature_dim)

2. lstm_input = [Emb(word_t) ‖ ctx]               (B, emb_dim+feature_dim)
3. h, c       = LSTMCell(lstm_input, (h,c))
4. logit_t    = FC(Dropout(h))                     (B, vocab_size)
```

#### Régularisation doubly stochastic (Xu et al. 2015)

Pour forcer l'attention à couvrir l'intégralité de la grille :

```
L_att = λ · mean((1 − Σ_t alpha[t, p])²)
L_total = L_ce + L_att
```

Valeur par défaut : `attention_lambda = 1.0`.

### Portes du LSTMCell

| Porte | Formule | Rôle |
|-------|---------|------|
| Forget `f_t` | `σ(W_f·[h,x] + b_f)` | Quoi oublier de la mémoire |
| Input `i_t` | `σ(W_i·[h,x] + b_i)` | Quoi retenir de l'entrée |
| Cell `c_t` | `f_t·c_{t-1} + i_t·tanh(W_c·[h,x])` | Mise à jour mémoire |
| Output `h_t` | `σ(W_o·[h,x])·tanh(c_t)` | Sortie filtrée |

### Génération — Greedy vs Beam Search

**Greedy :** à chaque step, choisir `argmax(logits)`.

**Beam search (défaut, beam_width=5) :**

```
beams = [(score=0, tokens=[START], hidden)]
pour chaque step :
    pour chaque beam :
        si tokens[-1] == END : compléter
        sinon : générer top-beam_width candidats
    garder les beam_width meilleures hypothèses (score cumulé)
→ choisir l'hypothèse : max(score / len(tokens))
```

---

## 6. Schedulers LR

Les deux schedulers partagent un **warmup linéaire** de `warmup_epochs=5` epochs.

### 6.1 CosineAnnealingWarmRestarts — `--scheduler cosine`

```
LR (log)
  │
  │   warmup              cycle 1 (T0=10)          cycle 2 (T1=20)
  │ /‾‾‾‾‾‾‾\__________/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\__________________________/‾‾‾‾
  │            restart↑                          restart↑
  └──────────────────────────────────────────────────────────── epochs
```

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `warmup_epochs` | 5 | Montée linéaire initiale |
| `cosine_T0` | 10 | Durée du premier cycle |
| `cosine_T_mult` | 2 | Doublement à chaque restart |
| `lr_min` | 1e-5 | LR plancher |
| `max_no_improve_cycles` | 3 | Cycles max sans amélioration val loss |

**Early stopping cosine :** déclenché si la meilleure val loss globale ne s'améliore pas sur `max_no_improve_cycles` fins de cycle consécutives.

### 6.2 ReduceLROnPlateau — `--scheduler plateau`

```
LR
  │────────────────────                     (patience=10 epochs stagnantes)
  │                    ╲──────────────      × factor=0.5
  │                               ╲──────  × 0.5
  └─────────────────────────────────────── epochs
```

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `plateau_patience` | 10 | Epochs sans amélioration avant réduction |
| `plateau_factor` | 0.5 | Multiplicateur LR |
| `lr_min` | 1e-5 | LR plancher |
| `patience` | 15 | Early stop : epochs max sans amélioration |

### 6.3 SWA — Stochastic Weight Averaging (cosine uniquement)

Implémenté dans `_average_cycle_checkpoints()` :

```
1. À chaque fin de cycle → copie best_model.pth → best_model_cycle_N.pth
2. Fin d'entraînement (si ≥ 2 cycles complétés) :
     W_avg = (1/N) × Σᵢ W_cycle_i   ← moyenne arithmétique paramètre par paramètre
3. Évaluation : val_loss(W_avg) calculée et comparée au meilleur cycle
4. Sauvegarde → averaged_model.pth
```

> **Pourquoi SWA fonctionne :** chaque cycle cosine converge vers un minimum légèrement différent dans l'espace des paramètres. La moyenne arithmétique atterrit typiquement dans une zone plus plate de la loss surface, ce qui améliore la généralisation à l'inférence.

---

## 7. Boucle d'entraînement

### Architecture du Trainer

```python
Trainer(
    model, train_loader, val_loader, vocabulary, config,
    scheduler_type='cosine',   # 'cosine' ou 'plateau'
    val_pairs=None             # pour CIDEr avec 5 refs/image
)
```

### Boucle principale (`train()`)

```
pour chaque epoch :
    1. train_epoch(epoch)
       ├── forward (avec ou sans forward_with_alphas)
       ├── loss = L_ce + λ·L_att (si attention)
       ├── backward + clip_grad_norm(max_norm=5)
       └── optimizer.step()

    2. validate()
       └── val_loss = CrossEntropy(ignore_index=PAD, label_smoothing=0.1)

    3. _step_scheduler(epoch, val_loss, warmup_epochs)
       ├── si epoch < warmup_epochs : LR linéaire
       ├── si plateau : scheduler.step(val_loss)
       └── si cosine  : scheduler.step(cosine_epoch) + détection fin de cycle

    4. compute_metrics() [tous les bleu_every epochs]
       └── BLEU-1/4, METEOR, CIDEr — INFORMATIFS uniquement

    5. sauvegarder si val_loss < best_val_loss → best_model.pth
       sauvegarder si CIDEr ≥ best → best_model_cider.pth [informatif]

    6. early_stopping()
       ├── plateau : patience_counter sur epochs
       └── cosine  : no_improve_cycles sur fins de cycle
           + si fin de cycle : sauvegarder best_model_cycle_N.pth

fin :
    si cosine et ≥ 2 cycles : _average_cycle_checkpoints() → averaged_model.pth
    plot_learning_curves() + save_history()
```

### Loss

```python
# CrossEntropy avec label smoothing
criterion = nn.CrossEntropyLoss(
    ignore_index=vocab.word2idx['<PAD>'],
    label_smoothing=0.1
)

# Régularisation attention (resnet/densenet)
if use_attention_reg:
    outputs, alphas = model.forward_with_alphas(images, inputs)
    attention_sum = alphas.sum(dim=1)                        # (B, num_pixels)
    L_att = ((1.0 - attention_sum) ** 2).mean()
    loss = L_ce + attention_lambda * L_att
```

---

## 8. Inférence et évaluation

### Chargement d'un modèle

```python
from models.caption_model import load_model

# Auto-détection de l'architecture depuis le checkpoint
model, info = load_model('checkpoints/densenet/cosine/averaged_model.pth', device='cuda')
vocab = info['vocab']     # vocabulaire intégré dans le checkpoint
```

Le checkpoint sauvegarde `attention_dim`, `growth_rate`, `compression`, `block_config` — aucune configuration manuelle nécessaire pour le rechargement.

### Génération

```python
# Via le modèle complet
caption_indices = model.generate_caption(
    image,                  # (1, 3, 224, 224)
    max_length=20,
    start_token=1, end_token=2,
    method='beam_search'    # ou 'greedy'
)
caption = vocab.denumericalize(caption_indices[0])

# Avec poids d'attention (resnet/densenet)
tokens, alphas = model.generate_caption_with_attention(
    image, method='beam_search'
)
# alphas : (T, 49) — carte 7×7 par mot généré
```

### Métriques d'évaluation

| Métrique | Description | Implémentation |
|----------|-------------|----------------|
| **BLEU-1** | Précision unigrams | `nltk.corpus_bleu` + SmoothingFunction |
| **BLEU-4** | Précision 4-grams | `nltk.corpus_bleu` |
| **METEOR** | Alignement sémantique (synonymes) | `nltk.meteor_score` |
| **CIDEr-D** | Cohérence + TF-IDF, 5 refs/image | Implémentation interne |

> **CIDEr-D nécessite un corpus d'au moins 2 images** pour que l'IDF soit non nul. Avec 1 seul document, tous les n-grammes ont IDF = log(2/2) = 0 → score nul.

### Visualisation de l'attention

Disponible pour `model='resnet'` et `model='densenet'` uniquement.

```
Pour chaque mot généré :
  1. alpha_t : (49,) → reshape (7, 7) → upscale PIL (224, 224)
  2. overlay sur l'image originale (alpha=0.5, cmap='jet')
  3. mots de liaison (stop words) : opacité réduite
  4. overlay moyen (mots de contenu) : visualise les régions les plus regardées
```

---

## 9. Commandes CLI

### Installation

```bash
pip install torch torchvision tqdm matplotlib nltk
```

### Préparation des données (une fois)

```bash
# Télécharger COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip *.zip -d data/coco/

# Construire le vocabulaire
python prepare_data.py
python prepare_data.py --freq_threshold 10   # vocab plus compact
```

### Entraînement — `train.py`

```bash
# DenseNet + cosine (recommandé)
python train.py --model densenet --scheduler cosine

# ResNet + plateau
python train.py --model resnet --scheduler plateau

# CNN + cosine, mode développement rapide (5 epochs)
python train.py --model cnn --scheduler cosine --fast

# Reprendre depuis un checkpoint
python train.py --model densenet --scheduler cosine \
    --resume checkpoints/densenet/cosine/best_model.pth
```

### Génération — `demo.py`

```bash
# Image unique (modèle densenet/cosine par défaut)
python demo.py --model densenet --image ImagesTest/dog.jpg

# Dossier entier
python demo.py --model densenet --image_dir ImagesTest/

# Checkpoint explicite (averaged_model.pth recommandé)
python demo.py \
    --checkpoint checkpoints/densenet/cosine/averaged_model.pth \
    --image_dir ImagesTest/

# Greedy search, dossier de sortie personnalisé
python demo.py --model resnet --image_dir ImagesTest/ \
    --method greedy --save_dir results/resnet_demo/
```

### Visualisation attention — `visualize_attention.py`

```bash
# Image unique
python visualize_attention.py --model densenet --image ImagesTest/dog.jpg

# Dossier complet avec greedy
python visualize_attention.py --model resnet \
    --image_dir ImagesTest/ --method greedy

# Avec stop words visibles dans la grille
python visualize_attention.py --model densenet \
    --image_dir ImagesTest/ --show_stop_words

# Checkpoint spécifique
python visualize_attention.py \
    --checkpoint checkpoints/densenet/cosine/averaged_model.pth \
    --image ImagesTest/dog.jpg
```

### Évaluation — `evaluate.py`

```bash
# Un modèle (toute la val = 5000 images)
python evaluate.py --model densenet --scheduler cosine

# Comparaison des 3 architectures
python evaluate.py --model densenet resnet cnn --scheduler cosine

# Rapide : 500 images, greedy
python evaluate.py --model densenet --num_samples 500 --method greedy

# Sauvegarder les captions générées
python evaluate.py --model densenet \
    --save_captions results/captions_densenet.json

# Checkpoint spécifique
python evaluate.py \
    --checkpoint checkpoints/densenet/cosine/averaged_model.pth
```

### Tests — `test.py`

```bash
# Tous les tests (70 tests, aucun dataset requis)
python test.py

# Mode verbeux
python test.py -v

# Un ou plusieurs groupes
python test.py TestEncoder
python test.py TestTrainer -v
python test.py TestEncoder TestDecoder TestCaptionModel

# Groupes disponibles :
# TestConfig · TestVocabulary · TestEncoder · TestDecoder
# TestCaptionModel · TestDataLoader · TestTrainer · TestDemo
# TestEvaluate · TestVisualizeAttention · TestIntegration
```

---

## 10. Hyperparamètres de référence

### Modèle

| Paramètre | CNN | ResNet | DenseNet |
|-----------|-----|--------|----------|
| `embedding_dim` | 256 | 256 | 256 |
| `hidden_dim` | 512 | 512 | 512 |
| `feature_dim` | 512 | 512 | 512 |
| `attention_dim` | — | 256 | 256 |
| `dropout` | 0.3 | 0.3 | 0.3 |
| `attention_lambda` | 0.0 | 1.0 | 1.0 |
| `growth_rate` | — | — | 32 |
| `block_config` | — | — | (6,12,24,16) |

### Entraînement

| Paramètre | Valeur | Notes |
|-----------|--------|-------|
| `learning_rate` | 0.0003 (CNN/ResNet), 0.0002 (DenseNet) | Adam |
| `weight_decay` | 1e-4 | L2 régularisation |
| `batch_size` | 32 | GPU ≥ 8 GB |
| `num_workers` | 4 | DataLoader |
| `warmup_epochs` | 5 | Commun aux 2 schedulers |
| `num_epochs` | 100 | Max |
| `save_every` | 5 | Checkpoint périodique |
| `bleu_every` | 1 | Calcul métriques (toutes les epochs) |
| `bleu_num_samples` | 5000 | Toute la val COCO → CIDEr fiable |
| `max_caption_length` | 20 | Génération |
| `beam_width` | 5 | Beam search |
| `freq_threshold` | 5 | Seuil vocabulaire |
| `image_size` | 224 | Taille images |

### Cosine scheduler

| Paramètre | Valeur |
|-----------|--------|
| `cosine_T0` | 10 |
| `cosine_T_mult` | 2 |
| `lr_min` | 1e-5 |
| `max_no_improve_cycles` | 3 |

### Plateau scheduler

| Paramètre | Valeur |
|-----------|--------|
| `plateau_patience` | 10 |
| `plateau_factor` | 0.5 |
| `lr_min` | 1e-5 |
| `patience` (early stop) | 15 |

---

## Ressources

- [Show and Tell (Vinyals et al. 2015)](https://arxiv.org/abs/1411.4555) — architecture de base encoder-decoder
- [Show, Attend and Tell (Xu et al. 2015)](https://arxiv.org/abs/1502.03044) — attention de Bahdanau + régularisation doubly stochastic
- [Densely Connected Networks (Huang et al. 2017)](https://arxiv.org/abs/1608.06993) — architecture DenseNet
- [Stochastic Weight Averaging (Izmailov et al. 2018)](https://arxiv.org/abs/1803.05407) — moyenne des poids de cycle
- [COCO Dataset](https://cocodataset.org) — dataset d'entraînement