"""
Configuration COCO
==================

encoder_type disponibles :
  'lite'      → EncoderCNNLite   + DecoderLSTM            (développement rapide)
  'full'      → EncoderCNN       + DecoderLSTM            (résiduel from scratch)
  'attention' → EncoderSpatial   + DecoderWithAttention   (résiduel + Bahdanau)
  'densenet'  → EncoderDenseNet  + DecoderWithAttention   (DenseNet-121 + Bahdanau)
                                                           ← recommandé sur COCO

Note sur num_layers :
  Ce paramètre ne s'applique qu'avec encoder_type='lite' ou 'full' (DecoderLSTM
  multi-couches). Avec 'attention' ou 'densenet', le decoder utilise un LSTMCell
  à une seule couche codée en dur — num_layers est ignoré.
"""

CONFIG = {
    # ========================================================================
    # CHEMINS
    # ========================================================================
    'data_dir': 'data',

    'train_captions_file': 'data/coco/annotations/captions_train2017.json',
    'val_captions_file':   'data/coco/annotations/captions_val2017.json',

    'train_images_dir': 'data/coco/train2017',
    'val_images_dir':   'data/coco/val2017',

    'vocab_path':      'data/coco_vocab.pkl',
    'checkpoint_dir':  'checkpoints_coco3',
    'log_dir':         'logs_coco3',
    'results_dir':     'results_coco3',

    # ========================================================================
    # HYPERPARAMÈTRES DU MODÈLE
    # ========================================================================

    'embedding_dim': 256,
    'hidden_dim':    512,
    'feature_dim':   512,

    # Ignoré avec encoder_type='attention' ou 'densenet' (LSTMCell fixé à 1 couche).
    # Actif uniquement avec 'lite' ou 'full'.
    'num_layers':    1,

    'dropout':       0.3,
    'attention_dim': 256,

    # ── Choix de l'architecture encoder ──────────────────────────────────────
    # 'densenet'  : DenseNet-121 from scratch + Bahdanau  ← recommandé
    # 'attention' : mini-ResNet from scratch  + Bahdanau  (ancienne option)
    # 'full'      : mini-ResNet from scratch  + LSTM global
    # 'lite'      : CNN minimal               + LSTM global  (dev rapide)
    'encoder_type':  'densenet',

    # ── Hyperparamètres DenseNet (ignorés si encoder_type != 'densenet') ─────
    # growth_rate  : k dans l'article — nombre de feature maps par couche
    #                32 = DenseNet-121 (bon compromis vitesse/qualité)
    #                48 = DenseNet-201 (plus lent, plus riche)
    'growth_rate':   32,

    # compression  : θ dans les couches de transition
    #                0.5 = compression par 2 entre chaque DenseBlock (standard)
    'compression':   0.5,

    # dense_dropout: dropout dans les DenseLayers
    #                0.0 recommandé — le dropout du decoder suffit généralement
    'dense_dropout': 0.0,

    # block_config : nombre de couches par DenseBlock
    #                (6, 12, 24, 16) = DenseNet-121  ~7.9 M params (encoder seul)
    #                (6, 12, 32, 32) = DenseNet-169  ~12.5 M params (plus lourd)
    'block_config':  (6, 12, 24, 16),

    # ========================================================================
    # HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # ========================================================================

    'num_epochs':    100,

    # Patience portée à 10 pour laisser le cosine scheduler compléter
    # au moins un cycle complet (T_0=10) avant de stopper.
    'patience':      10,

    'batch_size':    32,
    'warmup_epochs': 5,

    # LR légèrement réduit pour le DenseNet (plus profond = plus sensible au LR)
    'learning_rate': 0.0002,
    'weight_decay':  1e-4,
    'num_workers':   4,

    # ── Scheduler : CosineAnnealingWarmRestarts ───────────────────────────────
    # Remplace ReduceLROnPlateau(patience=2) qui gelait le LR trop tôt.
    # T_0=10 : premier cycle de 10 epochs après le warmup.
    # T_mult=2 : cycles suivants doublent → 10, 20, 40...
    # Le LR remonte à lr_target au début de chaque cycle.
    'cosine_T0':     10,
    'cosine_T_mult': 2,
    'lr_min':        1e-5,

    # ── Régularisation doubly stochastic (Xu et al. 2015) ────────────────────
    # Force l'attention à couvrir l'ensemble de la grille 7×7 plutôt que
    # de se concentrer sur les coins.
    # Pénalité : λ · mean((1 - Σ_t alpha[t, p])²)
    # 0.0 = désactivée (pour encoder_type='lite' ou 'full')
    # 1.0 = valeur recommandée par le papier original
    'attention_lambda': 1.0,

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    'image_size':      224,
    'freq_threshold':  5,
    'random_seed':     42,

    # ========================================================================
    # SAUVEGARDE
    # ========================================================================

    'save_every': 5,

    # ========================================================================
    # GÉNÉRATION
    # ========================================================================

    'max_caption_length': 20,
    'generation_method':  'beam_search',
    'beam_width':         5,

    # ========================================================================
    # MÉTRIQUES
    # ========================================================================

    'bleu_every':       2,

    # 5000 samples = toute la val COCO → CIDEr fiable (IDF corpus complet).
    # À 500 le CIDEr est trop bruité pour détecter une progression réelle.
    'bleu_num_samples': 5000,
}


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# ============================================================================

# Développement rapide
CONFIG_FAST = {
    **CONFIG,
    'num_epochs':       3,
    'encoder_type':     'lite',
    'batch_size':       64,
    'embedding_dim':    128,
    'hidden_dim':       256,
    'freq_threshold':   10,
    'attention_lambda': 0.0,
    'bleu_num_samples': 200,
}

# Résiduel from scratch (sans attention)
CONFIG_FULL = {
    **CONFIG,
    'num_epochs':       30,
    'encoder_type':     'full',
    'batch_size':       32,
    'attention_lambda': 0.0,
}

# Mini-ResNet + attention Bahdanau (ancienne configuration recommandée)
CONFIG_ATTENTION = {
    **CONFIG,
    'num_epochs':       100,
    'encoder_type':     'attention',
    'attention_dim':    256,
    'batch_size':       32,
    'learning_rate':    0.0003,
    'attention_lambda': 1.0,
}

# DenseNet-121 + attention Bahdanau (configuration recommandée)
CONFIG_DENSENET = {
    **CONFIG,
    'num_epochs':       100,
    'encoder_type':     'densenet',
    'growth_rate':      32,
    'compression':      0.5,
    'dense_dropout':    0.0,
    'block_config':     (6, 12, 24, 16),
    'batch_size':       32,
    'learning_rate':    0.0002,
    'attention_lambda': 1.0,
}

# DenseNet-169 + attention (plus puissant, plus lent)
CONFIG_DENSENET_169 = {
    **CONFIG_DENSENET,
    'block_config':  (6, 12, 32, 32),
    'batch_size':    16,             # plus lourd → batch réduit
    'learning_rate': 0.0001,
}

# GPU avec peu de mémoire
CONFIG_LOW_MEMORY = {
    **CONFIG,
    'batch_size':       8,
    'encoder_type':     'lite',
    'image_size':       128,
    'hidden_dim':       256,
    'embedding_dim':    128,
    'attention_lambda': 0.0,
    'bleu_num_samples': 200,
}