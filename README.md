# Image Captioning COCO

Projet de génération automatique de descriptions d'images entraîné sur COCO 2017,
avec trois architectures from scratch et deux schedulers configurables via CLI.

---

## Structure du projet

```
image_captioning/
│
├── config.py                 ← Configuration unifiée (toutes les archi / schedulers)
├── train.py                  ← Entraînement  (--model, --scheduler)
├── demo.py                   ← Génération de captions sur des images
├── evaluate.py               ← Calcul BLEU / METEOR / CIDEr sur val2017
├── visualize_attention.py    ← Cartes d'attention Bahdanau (resnet / densenet)
├── prepare_data.py           ← Construction du vocabulaire (à lancer 1 fois)
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
├── data/
│   ├── coco_vocab.pkl        ← Vocabulaire (généré par prepare_data.py)
│   └── coco/
│       ├── annotations/
│       │   ├── captions_train2017.json
│       │   └── captions_val2017.json
│       ├── train2017/
│       └── val2017/
│
├── checkpoints/
│   ├── cnn/      cosine/ plateau/
│   ├── resnet/   cosine/ plateau/
│   └── densenet/ cosine/ plateau/
│
├── logs/          ← Courbes d'entraînement (.png) et historiques (.json)
├── results/       ← Résultats d'évaluation (.json)
└── output_*/      ← Figures générées par demo.py et visualize_attention.py
```

---

## Architectures (`--model`)

| Modèle     | Encodeur                        | Décodeur                    | Notes                              |
|------------|---------------------------------|-----------------------------|------------------------------------|
| `cnn`      | EncoderCNN (résiduel, global)   | DecoderLSTM                 | Pas d'attention, plus rapide       |
| `resnet`   | EncoderSpatial (résiduel, 7×7)  | DecoderWithAttention        | Attention Bahdanau                 |
| `densenet` | EncoderDenseNet (DenseNet-121)  | DecoderWithAttention        | **Recommandé** – meilleur CIDEr    |

Tous les encodeurs sont **from scratch** (aucun poids pré-entraîné).

---

## Schedulers (`--scheduler`)

| Scheduler  | Comportement                                              | Early stopping                                     |
|------------|-----------------------------------------------------------|----------------------------------------------------|
| `plateau`  | `ReduceLROnPlateau(patience=10, factor=0.5)`              | Après `patience` epochs sans amélioration          |
| `cosine`   | `CosineAnnealingWarmRestarts(T0=10, T_mult=2)`            | Après `max_no_improve_cycles=3` cycles sans amélio |

Le scheduler `cosine` vérifie l'amélioration **à chaque fin de cycle**. Si 3 cycles
consécutifs ne produisent pas de progrès sur le CIDEr (ou val loss si CIDEr absent),
l'entraînement s'arrête.

Les deux schedulers sont précédés d'un **warmup linéaire** de 5 epochs.

---

## Installation

```bash
pip install torch torchvision tqdm matplotlib nltk
```

---

## Démarrage rapide

### 1. Télécharger COCO 2017

```bash
./getCOCO.sh
```

### 2. Préparer les données (1 seule fois)

```bash
python prepare_data.py
```

### 3. Entraîner

```bash
# DenseNet + cosine (recommandé)
python train.py --model densenet --scheduler cosine

# ResNet + plateau
python train.py --model resnet --scheduler plateau

# CNN basique + cosine
python train.py --model cnn --scheduler cosine

# Mode développement rapide (5 epochs, petit vocab)
python train.py --model densenet --scheduler cosine --fast

# Reprendre un entraînement
python train.py --model densenet --scheduler cosine --resume checkpoints/densenet/cosine/best_model.pth
```

### 4. Générer des captions

```bash
# Une seule image
python demo.py --model densenet --image ImagesTest/dog.jpg

# Tout un dossier
python demo.py --model densenet --image_dir ImagesTest/

# Avec greedy search
python demo.py --model resnet --image_dir ImagesTest/ --method greedy

# Checkpoint spécifique
python demo.py --checkpoint checkpoints/densenet/cosine/best_model_cider.pth --image ImagesTest/dog.jpg
```

### 5. Visualiser l'attention

```bash
# (uniquement resnet et densenet)
python visualize_attention.py --model densenet --image ImagesTest/dog.jpg
python visualize_attention.py --model resnet   --image_dir ImagesTest/
```

### 6. Évaluer

```bash
# Un modèle
python evaluate.py --model densenet --scheduler cosine

# Comparer plusieurs modèles
python evaluate.py --model densenet resnet cnn --scheduler cosine

# Évaluation rapide
python evaluate.py --model densenet --num_samples 500

# Sauvegarder les captions générées
python evaluate.py --model densenet --save_captions results/captions_densenet.json
```

---

## Checkpoints sauvegardés

Pour chaque combinaison `model/scheduler`, l'entraînement produit :

| Fichier                  | Contenu                                        |
|--------------------------|------------------------------------------------|
| `best_model.pth`         | Meilleur modèle selon la **val loss**          |
| `best_model_cider.pth`   | Meilleur modèle selon le **CIDEr**             |
| `checkpoint_epoch_N.pth` | Checkpoint périodique (tous les `save_every`)  |

Le vocabulaire est intégré dans chaque checkpoint → `load_model()` est autonome.

---

## Métriques

- **BLEU-1 / BLEU-4** : précision des n-grammes (NLTK)
- **METEOR** : alignement sémantique (NLTK)
- **CIDEr** : cohérence avec les 5 références humaines par image (implémentation interne)

La métrique principale surveillée pour l'early stopping et les checkpoints est le **CIDEr**.
En l'absence de CIDEr (calcul pas encore effectué), la **val loss** est utilisée.

---

## Configuration avancée

Modifier `config.py` pour ajuster les hyperparamètres :

```python
# Modifier CONFIG_DENSENET par exemple
CONFIG_DENSENET['learning_rate'] = 0.0001
CONFIG_DENSENET['batch_size']    = 16
CONFIG_DENSENET['cosine_T0']     = 20
```

Ou passer directement les valeurs dans `get_config()` depuis un script.