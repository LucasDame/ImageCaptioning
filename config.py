"""
config.py — Configuration unifiée pour Image Captioning COCO
=============================================================

Architectures disponibles (--model) :
  'cnn'       → EncoderCNN       + DecoderLSTM            (résiduel+CBAM from scratch, vecteur global)
  'resnet'    → EncoderSpatial   + DecoderWithAttention   (résiduel+CBAM from scratch, grille 14×14)
  'densenet'  → EncoderDenseNet  + DecoderWithAttention   (DenseNet-121+CBAM from scratch, grille 14×14)

Changements v4 par rapport à v3 :
  - grid_size      : 7  → 14  (196 patches au lieu de 49, +CBAM dans l'encodeur)
  - hidden_dim     : 512 → 1024 (plus de capacité LSTM, comme le modèle de référence)
  - learning_rate  : 0.0003 → 0.001 (scheduler patience 5→3 pour compenser)
  - plateau_patience: 5 → 3

Schedulers disponibles (--scheduler) :
  'plateau'   → ReduceLROnPlateau(patience=3, factor=0.5)
  'cosine'    → CosineAnnealingWarmRestarts avec vérification d'amélioration
                à chaque restart et early stop après 3 cycles sans amélioration
"""

# ============================================================================
# CONFIGURATION DE BASE
# ============================================================================

BASE_CONFIG = {
    # ── Chemins ──────────────────────────────────────────────────────────────
    'train_captions_file': 'data/coco/annotations/captions_train2017.json',
    'val_captions_file':   'data/coco/annotations/captions_val2017.json',
    'train_images_dir':    'data/coco/train2017',
    'val_images_dir':      'data/coco/val2017',
    'vocab_path':          'data/coco_vocab.pkl',
    'checkpoint_dir':      'checkpoints',
    'log_dir':             'logs',
    'results_dir':         'results',

    # ── Hyperparamètres du modèle ─────────────────────────────────────────────
    'embedding_dim': 256,
    'hidden_dim':    1024,   # v4 : 512 → 1024 (cohérent avec le modèle de référence)
    'feature_dim':   512,
    'attention_dim': 256,
    'dropout':       0.5,

    # ── Entraînement ─────────────────────────────────────────────────────────
    'num_epochs':    300,
    'batch_size':    32,
    'num_workers':   4,
    'learning_rate': 0.001,  # v4 : 0.0003 → 0.001 (cohérent avec le modèle de référence)
    'weight_decay':  1e-4,
    'warmup_epochs': 5,

    # ── Scheduler : ReduceLROnPlateau ─────────────────────────────────────────
    'plateau_patience': 3,   # v4 : 5 → 3 (LR réduit plus vite)
    'plateau_factor':   0.5,
    'lr_min':           1e-5,

    # ── Scheduler : CosineAnnealingWarmRestarts ───────────────────────────────
    'cosine_T0':               10,
    'cosine_T_mult':           2,
    'max_no_improve_cycles':   3,

    # ── Early stopping global ─────────────────────────────────────────────────
    'patience':      15,

    # ── Régularisation doubly stochastic (Xu et al. 2015) ─────────────────────
    # Activée automatiquement pour model='resnet' et model='densenet'.
    'attention_lambda': 1.0,

    # ── Preprocessing ─────────────────────────────────────────────────────────
    'image_size':     224,
    'freq_threshold': 5,
    'random_seed':    42,

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    'save_every': 5,

    # ── Génération ────────────────────────────────────────────────────────────
    'max_caption_length': 20,
    'generation_method':  'beam_search',
    'beam_width':         5,

    # ── Métriques ─────────────────────────────────────────────────────────────
    'bleu_every':       1,
    'bleu_num_samples': 5000,
}


# ============================================================================
# CONFIGURATIONS PAR ARCHITECTURE
# ============================================================================

CONFIG_CNN = {
    **BASE_CONFIG,
    'model':            'cnn',
    'learning_rate':    0.001,
    'attention_lambda': 0.0,   # pas d'attention → pénalité désactivée
    'checkpoint_dir':   'checkpoints/cnn',
    'log_dir':          'logs/cnn',
}

CONFIG_RESNET = {
    **BASE_CONFIG,
    'model':            'resnet',
    'learning_rate':    0.001,
    'attention_lambda': 1.0,
    'grid_size':        14,    # v4 : 196 patches (au lieu de 49)
    'checkpoint_dir':   'checkpoints/resnet',
    'log_dir':          'logs/resnet',
}

CONFIG_DENSENET = {
    **BASE_CONFIG,
    'model':            'densenet',
    'learning_rate':    0.001,
    'attention_lambda': 1.0,
    'grid_size':        14,    # v4 : 196 patches (au lieu de 49)
    'growth_rate':      32,
    'compression':      0.5,
    'dense_dropout':    0.0,
    'block_config':     (6, 12, 24, 16),
    'checkpoint_dir':   'checkpoints/densenet',
    'log_dir':          'logs/densenet',
}


# ============================================================================
# CONFIGURATIONS RAPIDES POUR LE DÉVELOPPEMENT
# ============================================================================

CONFIG_CNN_FAST = {
    **CONFIG_CNN,
    'num_epochs':       5,
    'batch_size':       64,
    'embedding_dim':    128,
    'hidden_dim':       256,
    'freq_threshold':   10,
    'bleu_num_samples': 200,
}

CONFIG_RESNET_FAST = {
    **CONFIG_RESNET,
    'num_epochs':       5,
    'batch_size':       32,
    'embedding_dim':    128,
    'hidden_dim':       256,
    'freq_threshold':   10,
    'bleu_num_samples': 200,
}

CONFIG_DENSENET_FAST = {
    **CONFIG_DENSENET,
    'num_epochs':       5,
    'batch_size':       16,
    'embedding_dim':    128,
    'hidden_dim':       256,
    'freq_threshold':   10,
    'bleu_num_samples': 200,
}

# Configuration GPU à mémoire limitée
CONFIG_LOW_MEMORY = {
    **BASE_CONFIG,
    'batch_size':       8,
    'model':            'cnn',
    'image_size':       128,
    'hidden_dim':       256,
    'embedding_dim':    128,
    'attention_lambda': 0.0,
    'bleu_num_samples': 200,
}


# ============================================================================
# TABLE DE LOOKUP
# ============================================================================

CONFIGS = {
    'cnn':            CONFIG_CNN,
    'resnet':         CONFIG_RESNET,
    'densenet':       CONFIG_DENSENET,
    'cnn_fast':       CONFIG_CNN_FAST,
    'resnet_fast':    CONFIG_RESNET_FAST,
    'densenet_fast':  CONFIG_DENSENET_FAST,
    'low_memory':     CONFIG_LOW_MEMORY,
}


def get_config(model: str, fast: bool = False) -> dict:
    """
    Retourne la config pour le modèle et le mode demandés.

    Args:
        model   : 'cnn', 'resnet' ou 'densenet'
        fast    : True pour utiliser la config de développement rapide

    Returns:
        dict : configuration complète (copie)
    """
    key = f'{model}_fast' if fast else model
    if key not in CONFIGS:
        raise ValueError(
            f"Modèle inconnu : '{key}'. "
            f"Valeurs acceptées : {list(CONFIGS.keys())}"
        )
    return dict(CONFIGS[key])