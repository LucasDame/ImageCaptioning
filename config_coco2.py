"""
Configuration COCO
==================

Version COCO du fichier de configuration.

encoder_type disponibles (nouveau modèle) :
  'lite'      → EncoderCNNLite  + DecoderLSTM           (développement rapide)
  'full'      → EncoderCNN      + DecoderLSTM           (résiduel from scratch)
  'attention' → EncoderSpatial  + DecoderWithAttention  (meilleure qualité)
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
    'num_layers':    1,           # Ignoré si encoder_type='attention' (LSTMCell)
    'dropout':       0.5,

    'encoder_type':  'attention',      # 'lite' | 'full' | 'attention'
                                  # 'attention' = résiduel + Bahdanau (meilleur)

    'attention_dim': 256,         # Dimension interne de l'attention
                                  # Utilisé uniquement si encoder_type='attention'

    # ========================================================================
    # HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # ========================================================================

    'num_epochs':    30,
    'patience':      5,
    'batch_size':    256,
    'learning_rate': 0.001,
    'weight_decay':  1e-5,
    'num_workers':   4,

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
    'generation_method':  'beam_search',  # 'greedy' ou 'beam_search'
    'beam_width':         3,
}


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# ============================================================================

# Développement rapide
CONFIG_FAST = {
    **CONFIG,
    'num_epochs':    3,
    'encoder_type':  'lite',
    'batch_size':    64,
    'embedding_dim': 128,
    'hidden_dim':    256,
    'freq_threshold': 10,
}

# Résiduel from scratch, beam search
CONFIG_FULL = {
    **CONFIG,
    'num_epochs':   30,
    'encoder_type': 'full',
    'batch_size':   32,
}

# Résiduel + attention Bahdanau (meilleure qualité)
CONFIG_ATTENTION = {
    **CONFIG,
    'num_epochs':    30,
    'encoder_type':  'attention',
    'attention_dim': 256,
    'batch_size':    32,
}

# GPU avec peu de mémoire
CONFIG_LOW_MEMORY = {
    **CONFIG,
    'batch_size':    8,
    'encoder_type':  'lite',
    'image_size':    128,
    'hidden_dim':    256,
    'embedding_dim': 128,
}
