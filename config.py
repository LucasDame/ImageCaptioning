"""
config.py — Configuration unifiée pour Image Captioning COCO
=============================================================

Architectures disponibles (--model) :
  'cnn'       → EncoderCNN       + DecoderLSTM            (résiduel from scratch, vecteur global)
  'resnet'    → EncoderSpatial   + DecoderWithAttention   (résiduel from scratch + Bahdanau)
  'densenet'  → EncoderDenseNet  + DecoderWithAttention   (DenseNet-121 from scratch + Bahdanau)

Schedulers disponibles (--scheduler) :
  'plateau'   → ReduceLROnPlateau(patience=10, factor=0.5)
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
    'hidden_dim':    512,
    'feature_dim':   512,
    'attention_dim': 256,
    'dropout':       0.3,

    # ── Entraînement ─────────────────────────────────────────────────────────
    'num_epochs':    300,
    'batch_size':    32,
    'num_workers':   4,
    'learning_rate': 0.0003,
    'weight_decay':  1e-4,
    'warmup_epochs': 5,

    # ── Scheduler : ReduceLROnPlateau ─────────────────────────────────────────
    # Utilisé si --scheduler plateau
    # patience=10 : attend 10 epochs sans amélioration avant de réduire le LR
    # factor=0.5  : divise le LR par 2 à chaque réduction
    # min_lr      : LR plancher
    'plateau_patience': 5,
    'plateau_factor':   0.5,
    'lr_min':           1e-5,

    # ── Scheduler : CosineAnnealingWarmRestarts ───────────────────────────────
    # Utilisé si --scheduler cosine
    # T0=10       : durée du premier cycle cosine (après le warmup)
    # T_mult=2    : chaque restart double la durée → 10, 20, 40 epochs...
    # max_no_improve_cycles=3 : early stop si 3 cycles consécutifs sans amélioration
    'cosine_T0':               10,
    'cosine_T_mult':           2,
    'max_no_improve_cycles':   3,

    # ── Early stopping global ─────────────────────────────────────────────────
    # Pour plateau  : arrêt si patience_counter >= patience epochs sans amélioration
    # Pour cosine   : arrêt si max_no_improve_cycles cycles sans amélioration
    # La métrique surveillée est CIDEr si disponible, sinon val loss.
    'patience':      15,

    # ── Régularisation doubly stochastic (Xu et al. 2015) ─────────────────────
    # Activée automatiquement pour model='resnet' et model='densenet'.
    # 0.0 = désactivée, 1.0 = valeur recommandée par le papier.
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
    # bleu_every=1    : calcul des métriques à chaque epoch
    # bleu_num_samples: nombre d'images val utilisées pour BLEU/CIDEr
    #   5000 = toute la val COCO → CIDEr fiable
    #   500  = rapide pour le dev (CIDEr bruité)
    'bleu_every':       1,
    'bleu_num_samples': 5000,
}


# ============================================================================
# CONFIGURATIONS PAR ARCHITECTURE
# ============================================================================
# Chaque CONFIG_* surcharge uniquement ce qui change par rapport à BASE_CONFIG.

CONFIG_CNN = {
    **BASE_CONFIG,
    'model':            'cnn',
    'learning_rate':    0.0003,
    'attention_lambda': 0.0,   # pas d'attention → pénalité désactivée
    'checkpoint_dir':   'checkpoints/cnn',
    'log_dir':          'logs/cnn',
}

CONFIG_RESNET = {
    **BASE_CONFIG,
    'model':            'resnet',
    'learning_rate':    0.0003,
    'attention_lambda': 1.0,
    'checkpoint_dir':   'checkpoints/resnet',
    'log_dir':          'logs/resnet',
}

CONFIG_DENSENET = {
    **BASE_CONFIG,
    'model':            'densenet',
    'learning_rate':    0.0002,   # légèrement réduit : DenseNet est plus profond
    'attention_lambda': 1.0,
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
    'batch_size':       64,
    'embedding_dim':    128,
    'hidden_dim':       256,
    'freq_threshold':   10,
    'bleu_num_samples': 200,
}

CONFIG_DENSENET_FAST = {
    **CONFIG_DENSENET,
    'num_epochs':       5,
    'batch_size':       32,
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
# TABLE DE LOOKUP — utilisée par train.py, demo.py, evaluate.py, etc.
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