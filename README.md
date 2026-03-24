# 🖼️ Image Captioning COCO

Génération automatique de descriptions d'images entraîné sur le dataset **COCO 2017**, avec trois architectures encodeur–décodeur construites **from scratch** et deux stratégies de scheduler configurables via CLI.

---

## Table des matières

1. [Structure du projet](#structure-du-projet)
2. [Vue d'ensemble](#vue-densemble)
3. [Pipeline](#pipeline)
4. [Architectures disponibles](#architectures-disponibles)
5. [Schedulers](#schedulers)
6. [Installation](#installation)
7. [Démarrage rapide](#démarrage-rapide)
8. [Commandes détaillées](#commandes-détaillées)
9. [Configuration avancée](#configuration-avancée)
10. [Métriques d'évaluation](#métriques-dévaluation)


---

## Structure du projet

```
ImageCaptioning/
│
├── config.py                 ← Configuration unifiée (toutes les archi / schedulers)
├── train.py                  ← Entraînement (--model, --scheduler)
├── demo.py                   ← Génération de captions sur des images
├── evaluate.py               ← Calcul BLEU / METEOR / CIDEr sur val2017
├── visualize_attention.py    ← Cartes d'attention Bahdanau (resnet / densenet)
├── prepare_data.py           ← Construction du vocabulaire (à lancer 1 fois)
├── test.py                   ← 70 tests unitaires (11 groupes)
├── getCOCO.sh                ← Script de téléchargement COCO 2017
├── requirements.txt
│
├── models/
│   ├── encoder.py            ← EncoderCNN · EncoderSpatial · EncoderDenseNet
│   ├── decoder.py            ← DecoderLSTM · DecoderWithAttention (Bahdanau)
│   └── caption_model.py      ← Factory · save_model · load_model
│
├── utils/
│   ├── vocabulary.py         ← Construction et sérialisation du vocabulaire
│   ├── preprocessing.py      ← ImagePreprocessor · CaptionPreprocessor (COCO JSON)
│   └── data_loader.py        ← ImageCaptionDataset · CaptionCollate · get_data_loaders
│
├── data/                     ← Généré par getCOCO.sh + prepare_data.py
│   ├── coco_vocab.pkl
│   └── coco/
│       ├── annotations/
│       │   ├── captions_train2017.json
│       │   └── captions_val2017.json
│       ├── train2017/
│       └── val2017/
│
├── checkpoints/              ← Générés à l'entraînement
│   ├── cnn/      cosine/ plateau/
│   ├── resnet/   cosine/ plateau/
│   └── densenet/ cosine/ plateau/
│
├── logs/                     ← Courbes (.png) et historiques (.json)
├── results/                  ← Résultats d'évaluation (.json)
├── ImagesTest/               ← Images de démonstration
└── docs/                     ← Documentation technique détaillée
```

---

## Vue d'ensemble

Ce projet implémente un système complet de **image captioning** (description automatique d'images) basé sur une architecture encodeur CNN + décodeur LSTM avec mécanisme d'attention de Bahdanau. L'ensemble du code est entraînable de zéro, sans poids pré-entraînés.

**Fonctionnalités principales :**
- 3 architectures d'encodeur : CNN résiduel, ResNet spatial, DenseNet-121
- Décodeur LSTM avec ou sans attention de Bahdanau
- 2 schedulers : `cosine` (CosineAnnealingWarmRestarts) et `plateau` (ReduceLROnPlateau)
- Warmup linéaire sur 5 epochs
- Early stopping automatique
- Génération par beam search ou greedy search
- Visualisation des cartes d'attention
- Évaluation BLEU-1/4, METEOR, CIDEr
- 70 tests unitaires

---

## Pipeline

```
Image (224×224×3)
        │
        ▼
┌───────────────┐
│   ENCODEUR    │   CNN / ResNet / DenseNet (from scratch)
│  feature_dim  │
└───────┬───────┘
        │ (B, 512) ou (B, 49, 512)
        ▼
┌───────────────┐
│   DÉCODEUR    │   LSTM + Attention de Bahdanau (optionnelle)
│   LSTM Cell   │
└───────┬───────┘
        │
        ▼
  Caption texte   "a dog running in the park"
```

---

## Architectures disponibles

Sélectionnées via l'argument `--model` :

| Modèle       | Encodeur                          | Décodeur                    | Attention  | Paramètres | Notes                       |
|--------------|-----------------------------------|-----------------------------|------------|------------|-----------------------------|
| `cnn`        | EncoderCNN (résiduel, global)     | DecoderLSTM                 | ✗          | ~7 M       | Le plus rapide              |
| `resnet`     | EncoderSpatial (résiduel, 7×7)    | DecoderWithAttention        | ✓ Bahdanau | ~10 M      | Grille spatiale 49 patches  |
| `densenet` ★ | EncoderDenseNet (DenseNet-121)    | DecoderWithAttention        | ✓ Bahdanau | ~18 M      | **Recommandé** — meilleur CIDEr |

> **Tous les encodeurs sont entraînés from scratch** — aucun poids pré-entraîné (ni ImageNet, ni autre).

---

## Schedulers

Sélectionnés via l'argument `--scheduler` :

| Scheduler  | Algorithme                         | Early stopping                              | Particularité                     |
|------------|------------------------------------|---------------------------------------------|-----------------------------------|
| `cosine`   | CosineAnnealingWarmRestarts        | Après `max_no_improve_cycles=3` cycles      | SWA — moyenne des poids de cycle  |
| `plateau`  | ReduceLROnPlateau(patience=5, ×0.5)| Après `patience=10` epochs sans amélioration| Réduction LR × 0.5 sur plateau    |

Les deux schedulers sont précédés d'un **warmup linéaire de 5 epochs**. La métrique surveillée pour l'early stopping est le **CIDEr** (ou la val loss si non disponible).

---

## Installation

**Prérequis :** python 3.8+, GPU recommandé (CUDA)

```bash
git clone https://github.com/LucasDame/ImageCaptioning.git
cd ImageCaptioning
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

**Dépendances :**
- `torch` / `torchvision`
- `tqdm`
- `matplotlib`
- `nltk`

---

## Démarrage rapide

### 1. Télécharger le dataset COCO 2017

```bash
./getCOCO.sh
```

Cela télécharge les images et annotations dans `data/coco/`.

### 2. Préparer le vocabulaire (une seule fois)

```bash
python3 prepare_data.py
```

Génère `data/coco_vocab.pkl`. Ne doit être lancé qu'une seule fois.

### 3. Entraîner un modèle

```bash
# Recommandé : DenseNet + cosine
python3 train.py --model densenet --scheduler cosine

# ResNet + plateau
python3 train.py --model resnet --scheduler plateau

# CNN basique + plateau
python3 train.py --model cnn --scheduler plateau

# Mode développement rapide (5 epochs, vocab réduit)
python3 train.py --model densenet --scheduler cosine --fast

# Reprendre un entraînement existant
python3 train.py --model densenet --scheduler cosine --resume checkpoints/densenet/cosine/best_model.pth
```

### 4. Générer des captions
```bash
# Sur une image unique
python3 demo.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image ImagesTest/dog.jpg

# Sur tout un dossier
python3 demo.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image_dir ImagesTest/

# Avec greedy search (par défaut : beam search)
python3 demo.py --checkpoint checkpoints/resnet/cosine/best_model.pth --method greedy

# Avec le meilleur CIDEr
python3 demo.py --checkpoint checkpoints/densenet/cosine/best_model_cider.pth --image ImagesTest/dog.jpg
```

### 5. Visualiser l'attention

Disponible uniquement pour les modèles avec attention (resnet et densenet) :
```bash
python3 visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image ImagesTest/dog.jpg
python3 visualize_attention.py --checkpoint checkpoints/resnet/cosine/best_model.pth --image_dir ImagesTest/
```

Génère des cartes d'attention superposées sur l'image, montrant les zones regardées mot par mot.

### 6. Évaluer un modèle
```bash
# Évaluation d'un checkpoint précis
python3 evaluate.py --checkpoint checkpoints/densenet/cosine/best_model_cider.pth

# Évaluation rapide sur 500 échantillons
python3 evaluate.py --checkpoint checkpoints/densenet/cosine/best_model.pth --num_samples 500

# Comparer plusieurs architectures (mode --model, requiert --scheduler)
python3 evaluate.py --model densenet resnet cnn --scheduler cosine

# Sauvegarder les captions générées
python3 evaluate.py --checkpoint checkpoints/densenet/cosine/best_model.pth --save_captions results/captions_densenet.json

# Evaluation d'un modèle par le chemin de son checkpoint
python3 evaluate.py --checkpoint checkpoints/densenet/cosine/best_model.pth
```

---

## Commandes détaillées

### `train.py`

| Argument       | Description                                        | Défaut       |
|----------------|----------------------------------------------------|--------------|
| `--model`      | Architecture : `cnn`, `resnet`, `densenet`         | Requis       |
| `--scheduler`  | Stratégie LR : `cosine`, `plateau`                 | Requis       |
| `--fast`       | Mode développement rapide (5 epochs, vocab réduit) | `False`      |
| `--resume`     | Chemin vers un checkpoint `.pth` pour reprendre    | `None`       |

### `demo.py`

| Argument       | Description                                         |
|----------------|-----------------------------------------------------|
| `--checkpoint` | **(Requis)** Chemin vers le fichier `.pth`          |
| `--image`      | Chemin vers une image unique                        |
| `--image_dir`  | Chemin vers un dossier d'images                     |
| `--method`     | Méthode de génération : `beam_search` ou `greedy`   |
| `--vocab_path` | Chemin vers le vocabulaire (si absent du checkpoint)|

### `evaluate.py`

Deux modes exclusifs :
- `--checkpoint` : évalue un fichier `.pth` précis
- `--model` + `--scheduler` : recherche automatiquement les meilleurs checkpoints

| Argument          | Description                                              |
|-------------------|----------------------------------------------------------|
| `--checkpoint`    | Chemin vers un fichier `.pth` (mode checkpoint direct)   |
| `--model`         | Un ou plusieurs modèles : `densenet resnet cnn` (requiert `--scheduler`) |
| `--scheduler`     | Scheduler utilisé à l'entraînement (requis avec `--model`) |
| `--num_samples`   | Nombre d'images val pour l'évaluation (défaut : 5000)    |
| `--save_captions` | Fichier de sortie JSON pour les captions générées        |

### `visualize_attention.py`

| Argument            | Description                                              |
|---------------------|----------------------------------------------------------|
| `--checkpoint`      | **(Requis)** Chemin vers le fichier `.pth` (resnet ou densenet uniquement) |
| `--image`           | Chemin vers une image unique                             |
| `--image_dir`       | Chemin vers un dossier d'images                         |
| `--method`          | Méthode de génération : `beam_search` ou `greedy`        |
| `--show_stop_words` | Inclure les mots de liaison dans la grille d'attention   |

---

## Configuration avancée

Tous les hyperparamètres sont centralisés dans `config.py`. Pour les modifier :

```python
# Exemple : modifier les hyperparamètres de DenseNet
CONFIG_DENSENET['learning_rate'] = 0.0001
CONFIG_DENSENET['batch_size']    = 16
CONFIG_DENSENET['cosine_T0']     = 20
```

**Hyperparamètres clés (BASE_CONFIG) :**

| Paramètre            | Valeur par défaut | Description                                      |
|----------------------|-------------------|--------------------------------------------------|
| `embedding_dim`      | 256               | Dimension des embeddings de mots                 |
| `hidden_dim`         | 512               | Dimension du hidden state LSTM                   |
| `attention_dim`      | 256               | Dimension de la couche d'attention               |
| `dropout`            | 0.3               | Taux de dropout                                  |
| `batch_size`         | 32                | Taille des batches                               |
| `learning_rate`      | 0.0003            | LR initial (0.0002 pour DenseNet)                |
| `num_epochs`         | 300               | Nombre maximum d'epochs                          |
| `beam_width`         | 5                 | Largeur du beam search                           |
| `max_caption_length` | 20                | Longueur maximale des captions générées          |
| `freq_threshold`     | 5                 | Fréquence minimale pour inclure un mot au vocab  |

**Configurations spéciales disponibles :**
- `CONFIG_CNN_FAST`, `CONFIG_RESNET_FAST`, `CONFIG_DENSENET_FAST` — développement rapide (5 epochs)
- `CONFIG_LOW_MEMORY` — GPU à mémoire limitée (batch_size=8, image_size=128)

---

## Métriques d'évaluation

| Métrique    | Description                                                   | Implémentation |
|-------------|---------------------------------------------------------------|----------------|
| **BLEU-1**  | Précision des unigrammes par rapport aux références           | NLTK           |
| **BLEU-4**  | Précision des 4-grammes (plus exigeant)                       | NLTK           |
| **METEOR**  | Alignement sémantique avec correspondance de synonymes        | NLTK           |
| **CIDEr**   | Cohérence avec les 5 descriptions humaines par image          | Interne        |

La **val loss** est la métrique principale pour les checkpoints et l'early stopping. Les métriques BLEU/METEOR/CIDEr sont calculées à titre informatif.

---

## Checkpoints sauvegardés

Pour chaque combinaison `model/scheduler`, l'entraînement produit dans `checkpoints/<model>/<scheduler>/` :

| Fichier                    | Contenu                                       |
|----------------------------|-----------------------------------------------|
| `best_model.pth`           | Meilleur modèle selon la **val loss**         |
| `best_model_cider.pth`     | Meilleur modèle selon le **CIDEr**            |
| `checkpoint_epoch_N.pth`   | Checkpoint périodique (tous les `save_every`) |

> Le vocabulaire est intégré dans chaque checkpoint — `load_model()` est entièrement autonome.

---

## Tests

Le projet inclut **70 tests unitaires** organisés en 11 groupes couvrant les encodeurs, décodeurs, data loaders, vocabulaire et pipeline complet :

```bash
python3 test.py
```

Pour exécuter un groupe spécifique :

```bash
python3 test.py TestConfig
python3 test.py TestVocabulary
python3 test.py TestEncoder
python3 test.py TestDecoder
python3 test.py TestCaptionModel
python3 test.py TestDataLoader
python3 test.py TestTrainer
python3 test.py TestDemo
python3 test.py TestEvaluate
python3 test.py TestIntegration
```

Pour éxecuter avec couverture de code :

```bash
coverage run test.py
coverage report -m
coverage html  # Génère un rapport HTML dans htmlcov/
```