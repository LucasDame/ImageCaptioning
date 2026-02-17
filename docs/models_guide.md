# Guide des Modèles - Image Captioning

## 📚 Vue d'ensemble

Ce guide explique l'architecture des modèles pour le projet d'image captioning.

## 🏗️ Architecture Globale

```
Image (224x224x3)
        ↓
   ENCODER CNN
        ↓
Feature Vector (512)
        ↓
   DECODER LSTM (avec embeddings)
        ↓
Caption (séquence de mots)
```

## 🔍 1. Encoder CNN (`encoder.py`)

### Rôle
Extraire les features visuelles d'une image et les transformer en un vecteur de taille fixe.

### Deux versions disponibles

#### EncoderCNN (Version complète)
- **Architecture**: 5 blocs convolutionnels
- **Paramètres**: ~15M
- **Meilleure qualité** mais plus lent à entraîner

```python
from models import EncoderCNN

encoder = EncoderCNN(feature_dim=512)
```

**Architecture détaillée**:
```
Input: (batch_size, 3, 224, 224)
  ↓
Bloc 1: Conv(3→64) + BN + ReLU + MaxPool → (64, 112, 112)
  ↓
Bloc 2: Conv(64→128) + BN + ReLU + MaxPool → (128, 56, 56)
  ↓
Bloc 3: 2×Conv(128→256) + BN + ReLU + MaxPool → (256, 28, 28)
  ↓
Bloc 4: 2×Conv(256→512) + BN + ReLU + MaxPool → (512, 14, 14)
  ↓
Bloc 5: 2×Conv(512→512) + BN + ReLU + MaxPool → (512, 7, 7)
  ↓
AdaptiveAvgPool → (512, 1, 1)
  ↓
Flatten + FC(512→512) + ReLU + Dropout
  ↓
Output: (batch_size, 512)
```

#### EncoderCNNLite (Version légère)
- **Architecture**: 4 blocs convolutionnels
- **Paramètres**: ~2M
- **Plus rapide** pour tester et développer

```python
from models import EncoderCNNLite

encoder = EncoderCNNLite(feature_dim=512)
```

**Architecture détaillée**:
```
Input: (batch_size, 3, 224, 224)
  ↓
Bloc 1: Conv(3→32) + BN + ReLU + MaxPool → (32, 56, 56)
  ↓
Bloc 2: Conv(32→64) + BN + ReLU + MaxPool → (64, 28, 28)
  ↓
Bloc 3: Conv(64→128) + BN + ReLU + MaxPool → (128, 14, 14)
  ↓
Bloc 4: Conv(128→256) + BN + ReLU + MaxPool → (256, 7, 7)
  ↓
AdaptiveAvgPool → (256, 1, 1)
  ↓
Flatten + FC(256→512) + ReLU + Dropout
  ↓
Output: (batch_size, 512)
```

### Composants clés

**Convolution (Conv2D)**:
- Extrait des features locales (bords, textures, formes)
- kernel_size=3: Petite fenêtre pour capturer les détails

**Batch Normalization (BN)**:
- Normalise les activations
- Accélère l'entraînement et améliore la stabilité

**ReLU**:
- Fonction d'activation: `f(x) = max(0, x)`
- Introduit la non-linéarité

**MaxPooling**:
- Réduit la dimension spatiale
- Garde les features les plus importantes

**AdaptiveAvgPool**:
- Force une taille de sortie fixe (1×1)
- Permet de gérer différentes tailles d'input

**Dropout**:
- Régularisation pour éviter l'overfitting
- Désactive aléatoirement 50% des neurones pendant l'entraînement

### Exemple d'utilisation

```python
import torch
from models import EncoderCNN

# Créer l'encoder
encoder = EncoderCNN(feature_dim=512)

# Image batch
images = torch.randn(4, 3, 224, 224)  # 4 images

# Forward pass
features = encoder(images)  # (4, 512)

print(f"Features shape: {features.shape}")
print(f"Nombre de paramètres: {encoder.get_num_params():,}")
```

## 🧠 2. Decoder LSTM (`decoder.py`)

### Rôle
Générer une caption mot par mot à partir des features de l'image.

### Architecture

```python
from models import DecoderLSTM

decoder = DecoderLSTM(
    feature_dim=512,      # Dimension des features de l'encoder
    embedding_dim=256,    # Dimension des word embeddings
    hidden_dim=512,       # Dimension du LSTM
    vocab_size=5000,      # Taille du vocabulaire
    num_layers=1,         # Nombre de couches LSTM
    dropout=0.5           # Taux de dropout
)
```

### Composants clés

#### 1. Embedding Layer
Convertit les indices de mots en vecteurs denses.

```python
# Vocabulaire
word2idx = {
    '<PAD>': 0,
    '<START>': 1,
    'dog': 45,
    'running': 123,
    '<END>': 2
}

# Embedding
embedding = nn.Embedding(
    num_embeddings=5000,  # Taille du vocabulaire
    embedding_dim=256,    # Dimension du vecteur
    padding_idx=0         # Index de <PAD>
)

# Utilisation
word_indices = torch.tensor([1, 45, 123, 2])  # [START, dog, running, END]
embedded = embedding(word_indices)  # (4, 256)
```

**Chaque mot devient un vecteur de 256 dimensions** que le LSTM peut traiter.

#### 2. Feature Projection
Projette les features de l'image dans l'espace du LSTM.

```python
feature_projection = nn.Linear(512, 512)  # feature_dim → hidden_dim
```

Les features deviennent le **hidden state initial** du LSTM.

#### 3. LSTM
Le cœur du decoder. Génère la séquence de mots.

```python
lstm = nn.LSTM(
    input_size=256,      # embedding_dim
    hidden_size=512,     # hidden_dim
    num_layers=1,
    batch_first=True
)
```

**Comment fonctionne le LSTM ?**

```
t=0: [START] → LSTM → prédit "a"
      ↑
  image features (hidden state initial)

t=1: "a" → LSTM → prédit "dog"
      ↑
  hidden state de t=0

t=2: "dog" → LSTM → prédit "is"
      ↑
  hidden state de t=1

t=3: "is" → LSTM → prédit "running"
...
```

Le LSTM garde une **mémoire** des mots précédents grâce à son hidden state.

#### 4. Output Layer
Projette le hidden state du LSTM vers le vocabulaire.

```python
fc = nn.Linear(hidden_dim, vocab_size)  # 512 → 5000
```

Pour chaque position, on obtient un **score pour chaque mot** du vocabulaire.

### Flow complet du Decoder

```
Image Features (512)
        ↓
Feature Projection → Hidden State Initial (512)
        ↓
Word "a" (index 45)
        ↓
Embedding Layer → Vector (256)
        ↓
LSTM (avec hidden state) → Output (512)
        ↓
FC Layer → Scores pour tous les mots (5000)
        ↓
Softmax → Probabilités
        ↓
Argmax → Mot prédit
```

### Teacher Forcing (Entraînement)

Pendant l'entraînement, on utilise **les vrais mots précédents**, pas les prédictions.

```python
# Caption: "a dog is running"
# Indices: [1, 45, 123, 67, 89, 2]
#          [START, a, dog, is, running, END]

# Input au decoder: [1, 45, 123, 67, 89]  (sans END)
# Target:           [45, 123, 67, 89, 2]  (sans START)

outputs = decoder(features, inputs)
loss = criterion(outputs, targets)
```

**Pourquoi ?** C'est plus stable et plus rapide que d'utiliser les prédictions.

### Génération (Inférence)

En inférence, on génère **mot par mot** en utilisant les prédictions précédentes.

```python
# Méthode greedy (simple)
caption = decoder.generate(
    features,
    max_length=20,
    start_token=1,
    end_token=2
)
```

**Algorithme**:
1. Commencer avec `<START>`
2. Prédire le mot le plus probable
3. Utiliser ce mot comme input pour l'étape suivante
4. Répéter jusqu'à `<END>` ou max_length

### Exemple d'utilisation

```python
import torch
from models import DecoderLSTM

# Créer le decoder
decoder = DecoderLSTM(
    feature_dim=512,
    embedding_dim=256,
    hidden_dim=512,
    vocab_size=5000,
    num_layers=1,
    dropout=0.5
)

# Features de l'encoder
features = torch.randn(4, 512)  # 4 images

# Captions (indices)
captions = torch.randint(0, 5000, (4, 15))  # 4 captions de longueur 15

# Forward pass (entraînement)
outputs = decoder(features, captions)  # (4, 15, 5000)

# Génération (inférence)
generated = decoder.generate(features[:1], max_length=20)  # (1, seq_len)
```

## 🎯 3. Modèle Complet (`caption_model.py`)

### ImageCaptioningModel

Combine l'encoder et le decoder.

```python
from models import create_model

model = create_model(
    vocab_size=5000,
    embedding_dim=256,
    hidden_dim=512,
    feature_dim=512,
    num_layers=1,
    dropout=0.5,
    encoder_type='lite'  # 'full' ou 'lite'
)
```

### Forward Pass (Entraînement)

```python
# Batch d'images et captions
images = torch.randn(4, 3, 224, 224)
captions = torch.randint(0, 5000, (4, 15))

# Forward
outputs = model(images, captions)  # (4, 15, 5000)

# Loss
inputs = captions[:, :-1]
targets = captions[:, 1:]
outputs_reshaped = outputs.reshape(-1, vocab_size)
targets_reshaped = targets.reshape(-1)
loss = criterion(outputs_reshaped, targets_reshaped)
```

### Génération de Caption

```python
# Image unique
image = torch.randn(1, 3, 224, 224)

# Générer
caption_indices = model.generate_caption(
    image,
    max_length=20,
    start_token=1,
    end_token=2,
    method='greedy'
)

# Convertir en texte
caption_text = vocabulary.denumericalize(caption_indices[0])
print(caption_text)  # "a dog is running in the park"
```

### Sauvegarde et Chargement

```python
from models import save_model, load_model

# Sauvegarder
save_model(
    model,
    'checkpoints/best_model.pth',
    optimizer=optimizer,
    epoch=10,
    loss=1.5,
    vocab=vocabulary
)

# Charger
model, info = load_model(
    'checkpoints/best_model.pth',
    device='cuda',
    encoder_type='lite'
)
```

## 📊 4. Choix des Hyperparamètres

### Dimensions

| Hyperparamètre | Petit | Moyen | Grand |
|----------------|-------|-------|-------|
| embedding_dim  | 128   | 256   | 512   |
| hidden_dim     | 256   | 512   | 1024  |
| feature_dim    | 256   | 512   | 1024  |
| num_layers     | 1     | 2     | 3     |

**Recommandé pour Flickr8k**: `embedding_dim=256`, `hidden_dim=512`, `feature_dim=512`, `num_layers=1`

### Régularisation

- **dropout**: 0.3-0.5 (0.5 recommandé)
- **weight_decay**: 1e-5 (dans l'optimizer)

### Trade-offs

**Plus de paramètres**:
- ✅ Meilleure capacité du modèle
- ✅ Peut apprendre des patterns complexes
- ❌ Risque d'overfitting
- ❌ Plus lent à entraîner
- ❌ Plus de mémoire GPU

**Moins de paramètres**:
- ✅ Plus rapide
- ✅ Moins de mémoire
- ✅ Moins d'overfitting
- ❌ Capacité limitée

## 🔬 5. Tests des Modèles

### Test de l'Encoder

```bash
python models/encoder.py
```

Sortie attendue:
```
Features output shape: torch.Size([4, 512])
Number of parameters: 2,XXX,XXX
```

### Test du Decoder

```bash
python models/decoder.py
```

Sortie attendue:
```
Outputs shape: torch.Size([4, 15, 5000])
Generated caption shape: torch.Size([1, seq_len])
```

### Test du Modèle Complet

```bash
python models/caption_model.py
```

## 📈 6. Complexité Computationnelle

### EncoderCNN (Lite)
- **Paramètres**: ~2M
- **FLOPs**: ~0.5G par image
- **Mémoire GPU**: ~500MB (batch_size=32)

### EncoderCNN (Full)
- **Paramètres**: ~15M
- **FLOPs**: ~3G par image
- **Mémoire GPU**: ~2GB (batch_size=32)

### DecoderLSTM
- **Paramètres**: ~5-10M (dépend de vocab_size)
- **Mémoire GPU**: ~1GB (batch_size=32)

### Total (Lite)
- **Paramètres**: ~7-12M
- **Mémoire GPU**: ~1.5-2GB (batch_size=32)
- **Temps d'entraînement**: ~2-3h pour 30 epochs (GPU moderne)

## ❓ FAQ

**Q: Quelle version de l'encoder utiliser ?**
A: `EncoderCNNLite` pour développer et tester rapidement. `EncoderCNN` pour les meilleurs résultats finaux.

**Q: Pourquoi utiliser LSTM au lieu de Transformer ?**
A: Les LSTM sont plus simples à implémenter from scratch et fonctionnent bien pour les séquences courtes (captions).

**Q: Combien de couches LSTM utiliser ?**
A: 1 couche suffit généralement. 2-3 couches peuvent améliorer légèrement mais augmentent l'overfitting.

**Q: Que faire si j'ai une erreur "out of memory" ?**
A: Réduire `batch_size`, utiliser `EncoderCNNLite`, ou réduire `hidden_dim`.

**Q: Les embeddings doivent-ils être pré-entraînés ?**
A: Non ! Le projet est from scratch. Les embeddings s'entraînent en même temps que le reste.

## 🔗 Ressources

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Show and Tell Paper](https://arxiv.org/abs/1411.4555)
- [PyTorch LSTM Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)