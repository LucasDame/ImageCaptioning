"""
Configuration COCO
==================

Version COCO du fichier de configuration.

encoder_type disponibles :
  'lite'      → EncoderCNNLite  + DecoderLSTM           (développement rapide)
  'full'      → EncoderCNN      + DecoderLSTM           (résiduel from scratch)
  'attention' → EncoderSpatial  + DecoderWithAttention  (meilleure qualité)

Correctifs v3 :
  - Scheduler : ReduceLROnPlateau → CosineAnnealingWarmRestarts
    (cosine_T0, cosine_T_mult, lr_min)
  - Régularisation doubly stochastic : attention_lambda (0 = désactivée)
  - bleu_num_samples porté à 2000 pour un CIDEr stable
  - patience porté à 10 pour laisser le scheduler compléter ses cycles
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
    'checkpoint_dir':  'checkpoints_coco2',
    'log_dir':         'logs_coco2',
    'results_dir':     'results_coco2',

    # ========================================================================
    # HYPERPARAMÈTRES DU MODÈLE
    # ========================================================================

    'embedding_dim': 256,
    'hidden_dim':    512,
    'feature_dim':   512,
    'num_layers':    1,
    'dropout':       0.3,

    'encoder_type':  'attention',
    'attention_dim': 256,

    # ========================================================================
    # HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # ========================================================================

    'num_epochs':    1000,

    # CORRECTIF : patience portée à 10 pour laisser le cosine scheduler
    # compléter au moins un cycle complet (T_0=10) avant de stopper.
    # L'ancienne valeur de 7 stoppait souvent en plein milieu d'un cycle.
    'patience':      10,

    'batch_size':    32,
    'warmup_epochs': 5,
    'learning_rate': 0.0003,
    'weight_decay':  1e-4,
    'num_workers':   4,

    # ── CORRECTIF scheduler ─────────────────────────────────────────────────
    # CosineAnnealingWarmRestarts remplace ReduceLROnPlateau(patience=2).
    #
    # T_0=10 : premier cycle cosine de 10 epochs (après le warmup).
    # T_mult=2 : chaque restart double la durée du cycle suivant → 10, 20, 40...
    # lr_min : LR plancher atteint en bas de chaque cycle.
    #
    # Le LR remonte à learning_rate au début de chaque cycle, permettant au
    # modèle d'explorer de nouveaux bassins de convergence.
    'cosine_T0':     10,
    'cosine_T_mult': 2,
    'lr_min':        1e-5,

    # ── CORRECTIF régularisation attention ──────────────────────────────────
    # Pénalité doubly stochastic (Xu et al. 2015) qui force l'attention à
    # couvrir toute la grille 7×7 au lieu de se concentrer sur les coins.
    # Formule : λ · mean((1 - Σ_t alpha[t, p])²)
    #
    # 0.0 = désactivée (pour encoder_type != 'attention')
    # 1.0 = valeur recommandée par le papier original
    # Augmenter si les coins dominent encore après quelques epochs.
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

    'bleu_every': 1,

    # CORRECTIF : porté de 500 à 2000.
    # À 500 samples, l'IDF du CIDEr est trop bruité → les variations
    # epoch-à-epoch reflètent l'échantillonnage plutôt que la qualité réelle.
    # À 2000 samples (4× plus), le signal devient exploitable.
    # Idéalement 5000 (toute la val) si le temps de calcul le permet.
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
    'attention_lambda': 0.0,   # pas d'attention en mode lite
    'bleu_num_samples': 500,   # rapide pour le dev
}

# Résiduel from scratch, beam search
CONFIG_FULL = {
    **CONFIG,
    'num_epochs':       30,
    'encoder_type':     'full',
    'batch_size':       32,
    'attention_lambda': 0.0,   # pas d'attention en mode full
}

# Résiduel + attention Bahdanau (meilleure qualité) — configuration principale
CONFIG_ATTENTION = {
    **CONFIG,
    'num_epochs':       100,
    'encoder_type':     'attention',
    'attention_dim':    256,
    'batch_size':       32,
    'attention_lambda': 1.0,
    'cosine_T0':        10,
    'cosine_T_mult':    2,
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