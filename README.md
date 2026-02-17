# Image Captioning from Scratch

## 📋 Description du Projet

Projet de Deep Learning consistant à développer un système de génération automatique de légendes pour des images, **entièrement from scratch avec PyTorch** (sans modèles pré-entraînés).

Le modèle utilise une architecture encoder-decoder :
- **Encoder (CNN)** : Extrait les caractéristiques visuelles de l'image
- **Decoder (LSTM)** : Génère la légende mot par mot à partir des features

## 🎯 Objectifs

- Implémenter une architecture complète d'image captioning sans utiliser de modèles pré-entraînés
- Entraîner le modèle sur un dataset standard (COCO, Flickr8k ou Flickr30k)
- Préparer une démo live pour la session d'examen finale
- Travailler en équipe de deux personnes

## 🗂️ Structure du Projet

```
image-captioning/
├── data/
│   ├── raw/                    # Données brutes
│   ├── processed/              # Données prétraitées
│   └── vocab.pkl              # Vocabulaire construit
├── models/
│   ├── encoder.py             # Architecture CNN
│   ├── decoder.py             # Architecture LSTM
│   └── caption_model.py       # Modèle complet
├── utils/
│   ├── data_loader.py         # Chargement des données
│   ├── vocabulary.py          # Construction du vocabulaire
│   └── preprocessing.py       # Prétraitement des images
├── train.py                   # Script d'entraînement
├── evaluate.py                # Script d'évaluation
├── demo.py                    # Script pour la démo
├── requirements.txt           # Dépendances
└── README.md                  # Ce fichier
```

## 🛠️ Technologies Utilisées

- **Python 3.9** : Langage de programmation
- **PyTorch** : Framework de deep learning
- **torchvision** : Manipulation d'images
- **NumPy** : Calculs numériques
- **Pillow** : Traitement d'images
- **Matplotlib** : Visualisation
- **NLTK** : Traitement du langage naturel

## 📊 Datasets Possibles

1. **Flickr8k** (recommandé pour débuter)
   - 8,000 images
   - 5 captions par image
   - Plus léger, entraînement plus rapide

2. **COCO**
   - 120,000+ images
   - 5 captions par image
   - Plus complexe, meilleurs résultats

## 🗺️ Feuille de Route

### Phase 1 : Préparation et Compréhension

#### ✅ Tâches à réaliser
- [x] Lire et comprendre l'architecture encoder-decoder
- [x] Étudier le fonctionnement des CNN et LSTM
- [x] Choisir le dataset 
- [x] Télécharger le dataset
- [x] Configurer l'environnement de développement
- [x] Installer les dépendances

---

### Phase 2 : Prétraitement des Données

#### ✅ Tâches à réaliser
- [x] Implémenter le chargement des images
- [x] Créer la classe `Vocabulary` pour construire le vocabulaire
- [x] Tokenizer les captions (ajout des tokens `<start>`, `<end>`, `<pad>`, `<unk>`)
- [x] Normaliser les images (resize, normalisation)
- [x] Créer le `DataLoader` PyTorch
- [x] Diviser les données (train/val/test)

#### 📝 Détails techniques
```python
# Tokens spéciaux
<start> : Début de séquence
<end>   : Fin de séquence
<pad>   : Padding
<unk>   : Mots inconnus
```

---

### Phase 3 : Implémentation de l'Encoder 

#### ✅ Tâches à réaliser
- [ ] Concevoir l'architecture CNN from scratch
- [ ] Implémenter les couches convolutionnelles
- [ ] Ajouter le pooling et la normalisation
- [ ] Créer la couche fully connected pour extraire le feature vector
- [ ] Tester l'encoder sur quelques images

#### 🏗️ Architecture suggérée
```
Input (224x224x3)
→ Conv2D(64) + ReLU + MaxPool
→ Conv2D(128) + ReLU + MaxPool
→ Conv2D(256) + ReLU + MaxPool
→ Conv2D(512) + ReLU + MaxPool
→ Flatten
→ Linear(2048) → Feature vector
```

---

### Phase 4 : Implémentation du Decoder 

#### ✅ Tâches à réaliser
- [ ] Implémenter la couche d'embedding pour les mots
- [ ] Créer l'architecture LSTM
- [ ] Implémenter la couche de sortie (softmax)
- [ ] Gérer les séquences de longueur variable
- [ ] Tester le decoder avec des features aléatoires

#### 🏗️ Architecture suggérée
```
Feature vector (2048)
→ Linear projection
→ Embedding layer pour les mots (W_emb)
→ LSTM cells (séquence)
→ Linear → Softmax (prédiction du prochain mot)
```

---

### Phase 5 : Assemblage du Modèle Complet

#### ✅ Tâches à réaliser
- [ ] Combiner encoder et decoder
- [ ] Implémenter la forward pass complète
- [ ] Définir la fonction de loss (CrossEntropyLoss)
- [ ] Configurer l'optimiseur (Adam recommandé)
- [ ] Tester sur un petit batch

#### 💡 Pipeline complet
```
Image → Encoder → Feature vector → Decoder → Caption
                                      ↑
                                 Previous words
```

---

### Phase 6 : Entraînement

#### ✅ Tâches à réaliser
- [ ] Implémenter la boucle d'entraînement
- [ ] Ajouter la validation après chaque epoch
- [ ] Implémenter le teacher forcing
- [ ] Sauvegarder les checkpoints
- [ ] Logger les métriques (loss, perplexity)
- [ ] Visualiser les courbes d'apprentissage
- [ ] Ajuster les hyperparamètres

#### ⚙️ Hyperparamètres à tester
- Learning rate : 0.001, 0.0001
- Batch size : 32, 64, 128
- Hidden size LSTM : 256, 512
- Embedding dimension : 256, 512
- Nombre d'epochs : 20-50

---

### Phase 7 : Évaluation et Amélioration

#### ✅ Tâches à réaliser
- [ ] Implémenter la génération de captions (beam search ou greedy)
- [ ] Calculer les métriques BLEU
- [ ] Analyser les résultats qualitatifs
- [ ] Identifier les cas d'échec
- [ ] Améliorer le modèle (data augmentation, dropout, etc.)

#### 📊 Métriques d'évaluation
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- Analyse visuelle des captions générées

---

### Phase 8 : Préparation de la Démo
#### ✅ Tâches à réaliser
- [ ] Créer un script de démo simple
- [ ] Tester avec plusieurs images
- [ ] Préparer une interface de visualisation
- [ ] Optimiser le temps d'inférence
- [ ] Préparer des exemples de succès et d'échecs
- [ ] Documenter les choix techniques

#### 🎬 Format de la démo
```python
# demo.py
1. Charger le modèle entraîné
2. Charger l'image fournie
3. Générer la caption
4. Afficher image + caption
```

---

### Phase 9 : Finalisation

#### ✅ Tâches à réaliser
- [ ] Nettoyer le code
- [ ] Ajouter des commentaires
- [ ] Finaliser le README
- [ ] Préparer les réponses aux questions potentielles
- [ ] Répéter la présentation
- [ ] Vérifier que tout fonctionne

#### 🎯 Points clés pour l'examen
- Comprendre chaque composant du modèle
- Savoir expliquer les choix d'architecture
- Être capable de discuter des résultats
- Connaître les limites du modèle

---

## 🚀 Installation et Utilisation

### Installation

```bash
# Cloner le repository
git clone <votre-repo>
cd image-captioning

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Entraînement

```bash
python train.py --data_path ./data/flickr8k \
                --epochs 30 \
                --batch_size 64 \
                --lr 0.001
```

### Évaluation

```bash
python evaluate.py --model_path ./checkpoints/best_model.pth \
                   --image_path ./test_images/
```

### Démo

```bash
python demo.py --model_path ./checkpoints/best_model.pth \
               --image_path ./exam_image.jpg
```

## 📈 Résultats Attendus

- **Loss** : Doit diminuer progressivement
- **BLEU-4** : > 0.15-0.20 pour un modèle from scratch sur Flickr8k
- **Qualité visuelle** : Captions cohérentes pour des images simples

## 🤝 Travail en Équipe

### Répartition suggérée des tâches

**Membre 1** :
- Prétraitement des données
- Implémentation de l'encoder
- Entraînement du modèle

**Membre 2** :
- Construction du vocabulaire
- Implémentation du decoder
- Évaluation et démo

**Ensemble** :
- Architecture globale
- Debugging
- Préparation de la présentation

## 📝 Questions Potentielles pour l'Examen

1. **Architecture**
   - Pourquoi utiliser un CNN pour l'encoder ?
   - Pourquoi un LSTM pour le decoder ?
   - Qu'est-ce que le teacher forcing ?

2. **Entraînement**
   - Quelle fonction de loss avez-vous utilisée ?
   - Comment gérez-vous les séquences de longueur variable ?
   - Quels sont vos hyperparamètres ?

3. **Résultats**
   - Quelles sont les performances de votre modèle ?
   - Quelles sont les limites ?
   - Comment pourriez-vous l'améliorer ?

## 🔧 Conseils Pratiques

1. **Commencez simple** : Testez sur un petit subset avant l'entraînement complet
2. **Sauvegardez régulièrement** : Checkpoints après chaque epoch
3. **Visualisez** : Regardez des exemples de captions pendant l'entraînement
4. **Débuggez progressivement** : Testez chaque composant séparément
5. **Documentez** : Notez tous vos choix et expérimentations

## 📚 Ressources Supplémentaires

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Show and Tell Paper](https://arxiv.org/abs/1411.4555)
- [COCO Dataset](https://cocodataset.org/)
- [BLEU Score Explanation](https://en.wikipedia.org/wiki/BLEU)

## 📄 Licence

Ce projet est réalisé dans le cadre d'un cours de Deep Learning.

## 👥 Auteurs

- [Votre Nom]
- [Nom de votre coéquipier]

---

**Bon courage pour votre projet ! 🎓**