# Guide du Preprocessing et DataLoader

## 📚 Vue d'ensemble

Ce guide explique comment utiliser les composants de preprocessing et de chargement des données pour le projet d'image captioning.

## 🗂️ Structure des fichiers

```
utils/
├── __init__.py           # Exports du module
├── vocabulary.py         # Construction et gestion du vocabulaire
├── preprocessing.py      # Preprocessing des images et captions
└── data_loader.py        # Dataset et DataLoader PyTorch
```

## 🔤 1. Vocabulaire (`vocabulary.py`)

### Qu'est-ce que c'est ?

Le vocabulaire est un dictionnaire qui fait le lien entre les mots et des indices numériques. C'est essentiel car les modèles de deep learning ne comprennent que des nombres.

### Tokens spéciaux

```python
<PAD>   (index 0): Padding pour égaliser les longueurs
<START> (index 1): Début de séquence
<END>   (index 2): Fin de séquence
<UNK>   (index 3): Mots inconnus (hors vocabulaire)
```

### Utilisation

```python
from utils.vocabulary import Vocabulary

# 1. Créer le vocabulaire
vocab = Vocabulary(freq_threshold=5)  # Mots avec freq < 5 seront <UNK>

# 2. Construire à partir des captions
captions = [
    "a dog is running",
    "two cats sitting",
    "a dog playing"
]
vocab.build_vocabulary(captions)

# 3. Convertir texte → indices
indices = vocab.numericalize("a dog is playing")
# Résultat: [1, 45, 123, 67, 89, 2]  (<START>, a, dog, is, playing, <END>)

# 4. Convertir indices → texte
text = vocab.denumericalize(indices)
# Résultat: "a dog is playing"

# 5. Sauvegarder/Charger
vocab.save("data/vocab.pkl")
vocab = Vocabulary.load("data/vocab.pkl")

# 6. Taille du vocabulaire (pour l'embedding layer)
vocab_size = len(vocab)
```

### Paramètres importants

- **freq_threshold**: Fréquence minimale pour inclure un mot
  - Trop bas (1-2): Vocabulaire très grand, risque d'overfitting
  - Trop haut (10+): Beaucoup de mots deviennent <UNK>
  - **Recommandé: 5** pour Flickr8k

## 🖼️ 2. Preprocessing (`preprocessing.py`)

### ImagePreprocessor

Transforme les images pour qu'elles soient utilisables par le CNN.

```python
from utils.preprocessing import ImagePreprocessor

# Créer le preprocessor
image_prep = ImagePreprocessor(
    image_size=224,      # Taille cible
    normalize=True       # Normalisation ImageNet
)

# Utiliser
image_tensor = image_prep("path/to/image.jpg", is_training=True)
# Résultat: Tensor de shape (3, 224, 224)
```

#### Transformations appliquées

**Mode entraînement** (`is_training=True`):
1. Resize à 224x224
4. Conversion en Tensor
5. Normalisation (mean et std d'ImageNet)

**Mode validation/test** (`is_training=False`):
1. Resize à 224x224
2. Conversion en Tensor
3. Normalisation

#### Pourquoi normaliser avec ImageNet ?

Les valeurs `mean=[0.485, 0.456, 0.406]` et `std=[0.229, 0.224, 0.225]` sont les statistiques du dataset ImageNet. C'est une pratique standard en vision par ordinateur.

### CaptionPreprocessor

Charge et organise les captions du dataset Flickr8k.

```python
from utils.preprocessing import CaptionPreprocessor

# Charger les captions
caption_prep = CaptionPreprocessor(
    captions_file="data/flickr8k/captions.txt",
    images_dir="data/flickr8k/Images"
)

# Obtenir toutes les captions (pour le vocabulaire)
all_captions = caption_prep.get_all_captions()

# Obtenir les paires image-caption
pairs = caption_prep.get_image_caption_pairs()
# Chaque pair: {'image_path': '...', 'caption': '...', 'image_name': '...'}

# Diviser en train/val/test
splits = caption_prep.split_data(train_ratio=0.8, val_ratio=0.1)
train_pairs = splits['train']
val_pairs = splits['val']
test_pairs = splits['test']
```

#### Format du fichier captions.txt

```
image1.jpg	a dog running in the park
image1.jpg	a brown dog is running
image2.jpg	two cats sitting on a wall
...
```

Chaque image peut avoir plusieurs captions (typiquement 5 dans Flickr8k).

## 📦 3. DataLoader (`data_loader.py`)

### Comprendre le flow des données

```
Fichier image.jpg + Caption texte
         ↓
ImageCaptionDataset
         ↓
DataLoader (avec batching et collate)
         ↓
Batch: (images, captions, lengths)
```

### ImageCaptionDataset

Dataset PyTorch custom pour l'image captioning.

```python
from utils.data_loader import ImageCaptionDataset

dataset = ImageCaptionDataset(
    image_caption_pairs=train_pairs,  # Liste de paires
    vocabulary=vocab,                 # Instance du vocabulaire
    image_preprocessor=image_prep,    # Preprocessor d'images
    is_training=True                  # Mode avec augmentation
)

# Utilisation
image_tensor, caption_tensor = dataset[0]
# image_tensor: (3, 224, 224)
# caption_tensor: (seq_len,) - indices des mots
```

### CaptionCollate

Fonction spéciale pour gérer le **padding** des captions de longueur variable.

#### Pourquoi le padding ?

Les captions ont des longueurs différentes, mais PyTorch nécessite des tensors de même taille dans un batch. Le padding résout ce problème en ajoutant des tokens `<PAD>` (index 0).

**Exemple:**
```
Caption 1: [1, 45, 123, 67, 2]           (longueur 5)
Caption 2: [1, 34, 56, 78, 90, 12, 2]    (longueur 7)

Après padding (max_len=7):
Caption 1: [1, 45, 123, 67, 2, 0, 0]     (paddée avec 0)
Caption 2: [1, 34, 56, 78, 90, 12, 2]    (inchangée)
```

### get_data_loaders

Fonction helper pour créer facilement les DataLoaders.

```python
from utils.data_loader import get_data_loaders

train_loader, val_loader = get_data_loaders(
    train_pairs=train_pairs,
    val_pairs=val_pairs,
    vocabulary=vocab,
    image_preprocessor=image_prep,
    batch_size=32,
    num_workers=4,
    shuffle_train=True
)

# Utiliser dans la boucle d'entraînement
for images, captions, lengths in train_loader:
    # images: (batch_size, 3, 224, 224)
    # captions: (batch_size, max_seq_len)
    # lengths: (batch_size,) - longueurs réelles
    pass
```

### Paramètres du DataLoader

- **batch_size**: Nombre d'images par batch
  - Trop petit (8-16): Entraînement lent, moins stable
  - Trop grand (128+): Risque d'out of memory
  - **Recommandé: 32-64**

- **num_workers**: Processus parallèles pour charger les données
  - 0: Un seul processus (plus lent)
  - 4-8: Bon compromis
  - Trop élevé: Overhead de gestion des processus

- **shuffle**: Mélanger les données
  - Train: **True** (essentiel pour la généralisation)
  - Val/Test: **False** (pas nécessaire)

- **pin_memory**: Accélère le transfert CPU→GPU
  - **True** si vous utilisez un GPU

## 🔄 4. Pipeline complet

### Étape par étape

```python
# 1. Charger les captions
from utils import CaptionPreprocessor, Vocabulary, ImagePreprocessor, get_data_loaders

caption_prep = CaptionPreprocessor(
    "data/flickr8k/captions.txt",
    "data/flickr8k/Images"
)

# 2. Construire le vocabulaire
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(caption_prep.get_all_captions())
vocab.save("data/vocab.pkl")

# 3. Diviser les données
splits = caption_prep.split_data(train_ratio=0.8, val_ratio=0.1)

# 4. Créer les DataLoaders
image_prep = ImagePreprocessor(image_size=224, normalize=True)
train_loader, val_loader = get_data_loaders(
    train_pairs=splits['train'],
    val_pairs=splits['val'],
    vocabulary=vocab,
    image_preprocessor=image_prep,
    batch_size=32,
    num_workers=4
)

# 5. Utiliser dans l'entraînement
for epoch in range(num_epochs):
    for images, captions, lengths in train_loader:
        # Votre code d'entraînement ici
        pass
```

### Script complet

Voir `prepare_data.py` pour un exemple complet fonctionnel.

## 🎓 5. Comprendre les Embeddings

### C'est quoi un embedding ?

Un embedding transforme un mot (représenté par un index) en un vecteur dense de nombres réels.

```
Mot "dog" (index 45) → Vecteur [0.23, -0.45, 0.12, ..., 0.67] (256 dimensions)
```

### Dans PyTorch

```python
import torch.nn as nn

# Créer l'embedding layer
vocab_size = len(vocab)        # Ex: 5000
embedding_dim = 256            # Dimension du vecteur

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=0  # L'embedding de <PAD> reste à 0
)

# Utiliser
word_indices = torch.tensor([1, 45, 123, 67, 2])  # [START, dog, is, running, END]
embedded = embedding(word_indices)
# Résultat: (5, 256) - 5 mots, chacun représenté par un vecteur de 256 dims
```

### Dans le Decoder

```python
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # 3. Couche de sortie
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, features, captions):
        # Les captions sont des indices (batch_size, seq_len)
        embeddings = self.embedding(captions)  # → (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeddings)     # → (batch_size, seq_len, hidden_dim)
        outputs = self.fc(lstm_out)             # → (batch_size, seq_len, vocab_size)
        return outputs
```

### Dimensions recommandées

- **Petit vocabulaire** (< 2000 mots): 128-256
- **Vocabulaire moyen** (2000-5000 mots): 256-512
- **Grand vocabulaire** (> 5000 mots): 512-1024

Pour Flickr8k avec ~2500 mots: **256 ou 512**

## ⚙️ 6. Format des données dans un batch

### Structure d'un batch

```python
images, captions, lengths = next(iter(train_loader))
```

#### `images`
- **Type**: `torch.Tensor`
- **Shape**: `(batch_size, 3, 224, 224)`
- **Contenu**: Images RGB normalisées
- **Range**: Environ [-2, 2] (après normalisation)

#### `captions`
- **Type**: `torch.Tensor`
- **Shape**: `(batch_size, max_seq_len)`
- **Contenu**: Indices des mots
- **Exemple**: 
  ```
  [[1, 45, 123, 67, 89, 2, 0, 0],    # Caption 1 (paddée)
   [1, 34, 56, 78, 2, 0, 0, 0],      # Caption 2 (paddée)
   [1, 90, 12, 34, 56, 78, 90, 2]]   # Caption 3 (complète)
  ```

#### `lengths`
- **Type**: `torch.Tensor`
- **Shape**: `(batch_size,)`
- **Contenu**: Longueur réelle de chaque caption (avant padding)
- **Exemple**: `[6, 5, 8]`

### Pourquoi garder les longueurs ?

Pour ignorer le padding lors du calcul de la loss:

```python
# Sans masking (mauvais - le padding affecte la loss)
loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

# Avec masking (bon - le padding est ignoré)
mask = (targets != pad_idx).float()
loss = (criterion_unreduced(outputs, targets) * mask).sum() / mask.sum()
```

## 🐛 7. Debugging et tests

### Vérifier le vocabulaire

```python
print(f"Taille: {len(vocab)}")
print(f"<PAD> index: {vocab.word2idx[vocab.pad_token]}")
print(f"<START> index: {vocab.word2idx[vocab.start_token]}")

# Tester la conversion
test = vocab.numericalize("a dog")
print(vocab.denumericalize(test))  # Doit retourner "a dog"
```

### Vérifier le DataLoader

```python
images, captions, lengths = next(iter(train_loader))
print(f"Images: {images.shape}")
print(f"Captions: {captions.shape}")
print(f"Lengths: {lengths.shape}")
print(f"\nPremière caption: {vocab.denumericalize(captions[0])}")
```

### Vérifier les transformations d'images

```python
import matplotlib.pyplot as plt

image_tensor = image_prep("test.jpg", is_training=False)

# Dénormaliser pour afficher
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img = image_tensor * std + mean
img = img.permute(1, 2, 0).numpy()

plt.imshow(img)
plt.show()
```

## 📊 8. Statistiques typiques pour Flickr8k

- **Nombre d'images**: 8,000
- **Captions par image**: 5
- **Total de captions**: 40,000
- **Taille du vocabulaire** (freq_threshold=5): ~2,500 mots
- **Longueur moyenne des captions**: 10-12 mots
- **Longueur max**: ~20-25 mots

## ❓ FAQ

**Q: Pourquoi utiliser teacher forcing ?**
A: Pendant l'entraînement, on donne les vrais mots précédents au modèle, pas ses prédictions. C'est plus stable et plus rapide.

**Q: Comment gérer les images de tailles différentes ?**
A: Le `ImagePreprocessor` les resize toutes à 224x224.

**Q: Que faire si je manque de mémoire GPU ?**
A: Réduire `batch_size` ou `image_size`.

**Q: Combien de temps pour charger les données ?**
A: Avec `num_workers=4`, environ instantané. Sans parallélisation, peut prendre quelques secondes par batch.

**Q: Dois-je normaliser les images ?**
A: **Oui**, c'est une bonne pratique qui aide à la convergence.

## 📝 Checklist

- [x] Dataset Flickr8k téléchargé
- [ ] Vocabulaire construit et sauvegardé
- [ ] DataLoaders testés avec un batch
- [ ] Comprendre le format des données
- [ ] Comprendre le padding et le collate
- [ ] Comprendre les embeddings
- [ ] Prêt à implémenter l'encoder et le decoder !
