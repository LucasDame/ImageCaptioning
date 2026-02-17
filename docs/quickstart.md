# 🚀 Guide de Démarrage Rapide

## Installation et Premier Entraînement

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Télécharger le dataset Flickr8k

Structure attendue:
```
data/
└── flickr8k/
    ├── Images/           # Toutes les images
    └── captions.txt      # Fichier de captions
```

Format de `captions.txt`:
```
image1.jpg	a dog running in the park
image1.jpg	a brown dog is running
image2.jpg	two cats sitting on a wall
...
```

### 3. Préparer les données (optionnel mais recommandé)

```bash
python prepare_data.py
```

Ce script va:
- Charger les captions
- Construire le vocabulaire
- Créer les splits train/val/test
- Tester un batch
- Sauvegarder le vocabulaire dans `data/vocab.pkl`

### 4. Entraîner le modèle

```bash
python train.py
```

**Configuration par défaut** (dans `train.py`):
- Encoder: `lite` (rapide)
- Epochs: 30
- Batch size: 32
- Learning rate: 0.001
- Embedding dim: 256
- Hidden dim: 512

**Résultats**:
- Modèles sauvegardés dans `checkpoints/`
- Meilleur modèle: `checkpoints/best_model.pth`
- Courbes d'apprentissage: `logs/learning_curves.png`
- Historique: `logs/training_history.json`

### 5. Évaluer le modèle

```bash
python evaluate.py
```

Calcule les métriques BLEU sur le test set et génère:
- `results/evaluation_results.json` (scores BLEU)
- `results/caption_examples.json` (exemples de captions)

### 6. Tester avec une image (Démo)

```bash
# Image unique
python demo.py --image data/test_image.jpg

# Avec sauvegarde du résultat
python demo.py --image data/test_image.jpg --output results/demo.png

# Batch d'images
python demo.py --image data/test_images/ --batch --output results/batch/
```

---

## 📁 Structure du Projet

```
image-captioning/
│
├── data/
│   ├── flickr8k/
│   │   ├── Images/              # Images du dataset
│   │   └── captions.txt         # Fichier de captions
│   └── vocab.pkl                # Vocabulaire (généré)
│
├── utils/
│   ├── __init__.py
│   ├── vocabulary.py            # Gestion du vocabulaire
│   ├── preprocessing.py         # Preprocessing images/captions
│   └── data_loader.py           # DataLoader PyTorch
│
├── models/
│   ├── __init__.py
│   ├── encoder.py               # CNN encoder
│   ├── decoder.py               # LSTM decoder
│   └── caption_model.py         # Modèle complet
│
├── docs/
│   ├── EMBEDDING_GUIDE.py       # Guide sur les embeddings
│   ├── PREPROCESSING_GUIDE.md   # Guide du preprocessing
│   └── MODELS_GUIDE.md          # Guide des modèles
│
├── checkpoints/                 # Modèles sauvegardés (généré)
├── logs/                        # Logs d'entraînement (généré)
├── results/                     # Résultats d'évaluation (généré)
│
├── prepare_data.py              # Script de préparation
├── train.py                     # Script d'entraînement
├── evaluate.py                  # Script d'évaluation
├── demo.py                      # Script de démo
├── requirements.txt             # Dépendances
└── README.md                    # Documentation principale
```

---

## 🎯 Workflow Typique

### Phase 1: Développement

1. **Tester le preprocessing**
   ```bash
   python prepare_data.py
   ```

2. **Entraînement rapide** (pour tester)
   ```python
   # Modifier train.py:
   config = {
       'num_epochs': 5,          # Juste 5 epochs
       'encoder_type': 'lite',   # Encoder léger
       'batch_size': 64,         # Plus grand batch
   }
   ```

3. **Vérifier que tout fonctionne**
   ```bash
   python train.py
   python evaluate.py
   python demo.py --image test.jpg
   ```

### Phase 2: Entraînement Final

1. **Configuration optimale**
   ```python
   config = {
       'num_epochs': 30-50,      # Plus d'epochs
       'encoder_type': 'lite',   # Ou 'full' si GPU puissant
       'batch_size': 32,
       'learning_rate': 0.001,
   }
   ```

2. **Lancer l'entraînement**
   ```bash
   python train.py
   ```
   
   ⏱️ Temps estimé:
   - Lite encoder: ~2-3h (GPU moderne)
   - Full encoder: ~6-8h (GPU moderne)

3. **Surveiller les courbes**
   - Regarder `logs/learning_curves.png` après chaque epoch
   - Train loss et val loss doivent descendre
   - Si val loss augmente → overfitting

### Phase 3: Évaluation et Démo

1. **Évaluer sur le test set**
   ```bash
   python evaluate.py
   ```
   
   Scores BLEU attendus (Flickr8k, from scratch):
   - BLEU-1: 0.50-0.60
   - BLEU-2: 0.30-0.40
   - BLEU-3: 0.20-0.25
   - BLEU-4: 0.15-0.20

2. **Préparer la démo pour l'examen**
   ```bash
   # Tester avec l'image fournie
   python demo.py --image exam_image.jpg --output results/exam_demo.png
   ```

---

## 🛠️ Modifier la Configuration

### Changer les hyperparamètres

Éditer `train.py`:

```python
config = {
    # Modèle
    'embedding_dim': 256,      # ↑ pour plus de capacité
    'hidden_dim': 512,         # ↑ pour plus de capacité
    'num_layers': 1,           # ↑ pour LSTM plus profond
    'dropout': 0.5,            # ↑ si overfitting
    
    # Entraînement
    'num_epochs': 30,          # ↑ pour converger
    'batch_size': 32,          # ↓ si out of memory
    'learning_rate': 0.001,    # ↓ si loss instable
    
    # Encoder
    'encoder_type': 'lite',    # 'full' pour meilleure qualité
}
```

### Utiliser l'encoder complet

```python
config['encoder_type'] = 'full'
```

⚠️ Nécessite plus de mémoire GPU

### Changer la taille des images

```python
config['image_size'] = 224  # Ou 128 pour plus rapide
```

---

## 🐛 Résolution de Problèmes

### Out of Memory (GPU)

**Solutions**:
1. Réduire `batch_size`: 32 → 16 → 8
2. Utiliser `encoder_type='lite'`
3. Réduire `image_size`: 224 → 128
4. Réduire `hidden_dim`: 512 → 256

### Loss ne descend pas

**Vérifications**:
1. Le vocabulaire est-il construit ? (`data/vocab.pkl` existe ?)
2. Les données sont-elles chargées ? (pas d'erreur dans `prepare_data.py` ?)
3. Le learning rate est-il trop bas ? (essayer 0.001)
4. Y a-t-il assez de données ? (Flickr8k = 8000 images)

### Loss explose (NaN)

**Solutions**:
1. Réduire le learning rate: 0.001 → 0.0001
2. Vérifier que gradient clipping est activé (déjà dans `train.py`)
3. Vérifier les données (pas de valeurs NaN)

### Captions générées sont bizarres

**Normal si**:
- Début d'entraînement (< 5 epochs)
- Modèle pas assez entraîné
- Données insuffisantes

**Solutions**:
- Entraîner plus longtemps (30+ epochs)
- Vérifier que val loss descend
- Augmenter la capacité du modèle

---

## 📊 Interpréter les Résultats

### Courbes d'apprentissage

```
Loss
 │
 │ ╲
 │  ╲  Train Loss
 │   ╲___________
 │
 │    ╲
 │     ╲  Val Loss
 │      ╲_________
 │
 └──────────────── Epochs
```

**Bon signe**: Les deux descendent et convergent

**Overfitting**: Train loss descend mais val loss augmente
→ Augmenter dropout, réduire num_layers

**Underfitting**: Les deux sont hautes et stables
→ Augmenter capacité du modèle, entraîner plus

### Scores BLEU

- **BLEU-1**: Compte les mots individuels (facile)
- **BLEU-2**: Compte les paires de mots (plus dur)
- **BLEU-3**: Compte les triplets (encore plus dur)
- **BLEU-4**: Compte les 4-grammes (le plus dur)

**Interprétation**:
- BLEU-4 > 0.20 → Excellent (rare from scratch)
- BLEU-4 > 0.15 → Très bon
- BLEU-4 > 0.10 → Bon
- BLEU-4 < 0.10 → À améliorer

---

## 🎓 Pour l'Examen Final

### Checklist de préparation

- [ ] Modèle entraîné (30+ epochs)
- [ ] Meilleur checkpoint sauvegardé
- [ ] Script de démo testé
- [ ] Image de test prête
- [ ] Comprendre l'architecture
- [ ] Pouvoir expliquer les choix

### Questions Potentielles

**Architecture**:
- Pourquoi CNN pour l'encoder ?
- Pourquoi LSTM pour le decoder ?
- Qu'est-ce que le teacher forcing ?
- Comment fonctionnent les embeddings ?

**Entraînement**:
- Quelle loss avez-vous utilisée ?
- Quels sont vos hyperparamètres ?
- Comment gérez-vous le padding ?
- Combien de temps pour entraîner ?

**Résultats**:
- Quels sont vos scores BLEU ?
- Montrer des exemples de succès/échecs
- Limites du modèle ?
- Comment améliorer ?

### Démo Live

```bash
# Script simple pour la démo
python demo.py --image <image_fournie.jpg> --output results/demo_final.png
```

**Timing**: < 5 secondes pour générer une caption

---

## 💡 Conseils

### Pour gagner du temps

1. **Utilisez `EncoderCNNLite`** pendant le développement
2. **Testez sur un subset** des données d'abord
3. **Sauvegardez régulièrement** les checkpoints
4. **Utilisez un GPU** si disponible

### Pour de meilleurs résultats

1. **Entraînez plus longtemps** (30-50 epochs)
2. **Augmentez la capacité** (hidden_dim, embedding_dim)
3. **Essayez différents learning rates**
4. **Ajustez le dropout** selon l'overfitting

### Pour la présentation

1. **Préparez plusieurs exemples** (bons et moins bons)
2. **Expliquez vos choix** (pourquoi ces hyperparamètres ?)
3. **Montrez les courbes** d'apprentissage
4. **Soyez honnête** sur les limites

---

## 🔗 Documentation Complète

- **README.md**: Vue d'ensemble du projet
- **PREPROCESSING_GUIDE.md**: Guide du preprocessing
- **EMBEDDING_GUIDE.py**: Guide des embeddings
- **MODELS_GUIDE.md**: Guide des modèles

Bon courage ! 🎓