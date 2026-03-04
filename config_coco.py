"""
Configuration COCO
==================

Version COCO du fichier de configuration.
"""

CONFIG = {
    # ========================================================================
    # CHEMINS
    # ========================================================================
    'data_dir': 'data',

    # Annotations COCO (format JSON officiel)
    'train_captions_file': 'data/coco/annotations/captions_train2017.json',
    'val_captions_file':   'data/coco/annotations/captions_val2017.json',

    # Dossiers d'images COCO (splits officiels)
    'train_images_dir': 'data/coco/train2017',
    'val_images_dir':   'data/coco/val2017',

    'vocab_path':       'data/coco_vocab.pkl',
    'checkpoint_dir':   'checkpoints_coco',
    'log_dir':          'logs_coco',
    'results_dir':      'results_coco',

    # ========================================================================
    # HYPERPARAMÈTRES DU MODÈLE
    # ========================================================================

    # Dimensions
    'embedding_dim': 256,       # Dimension des word embeddings
                                # Recommandé: 128-512

    'hidden_dim': 512,          # Dimension du LSTM hidden state
                                # Recommandé: 256-1024

    'feature_dim': 512,         # Dimension des features de l'encoder
                                # Standard: 512

    'num_layers': 1,            # Nombre de couches LSTM

    'dropout': 0.5,             # Taux de dropout

    'encoder_type': 'lite',     # 'lite' (~2M params) ou 'full' (~15M params)

    # ========================================================================
    # HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # ========================================================================

    'num_epochs': 30,           # COCO est ~15x plus grand que Flickr8k :
                                # 30 epochs suffisent en général

    'patience': 5,              # Patience pour l'early stopping

    'batch_size': 32,           # 32 pour GPU standard
                                # 64 si grosse GPU (>= 16 Go)
                                # 8-16 si peu de mémoire

    'learning_rate': 0.001,     # Standard Adam

    'weight_decay': 1e-5,       # Régularisation L2

    'num_workers': 4,           # Workers DataLoader

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    'image_size': 224,          # Taille des images

    'freq_threshold': 5,        # COCO a un vocabulaire beaucoup plus riche,
                                # freq_threshold=5 donne ~10 000 mots
                                # Monter à 10 pour réduire le vocab

    'random_seed': 42,          # Seed reproductibilité
                                # Note: pas de split_data() pour COCO,
                                # on utilise les splits officiels train2017/val2017

    # ========================================================================
    # SAUVEGARDE
    # ========================================================================

    'save_every': 5,            # Checkpoint tous les N epochs

    # ========================================================================
    # GÉNÉRATION (DÉMO/ÉVALUATION)
    # ========================================================================

    'max_caption_length': 20,
    'generation_method': 'greedy',  # 'greedy' ou 'beam_search'
}


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# ============================================================================

# Configuration pour un test rapide
CONFIG_FAST = {
    **CONFIG,
    'num_epochs': 3,
    'encoder_type': 'lite',
    'batch_size': 64,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'freq_threshold': 10,   # Vocab plus petit pour aller vite
}

# Configuration pour la qualité maximale
CONFIG_BEST = {
    **CONFIG,
    'num_epochs': 30,
    'encoder_type': 'full',
    'batch_size': 32,
    'embedding_dim': 512,
    'hidden_dim': 1024,
    'num_layers': 2,
    'freq_threshold': 5,
}

# Configuration pour GPU avec peu de mémoire
CONFIG_LOW_MEMORY = {
    **CONFIG,
    'batch_size': 8,
    'encoder_type': 'lite',
    'image_size': 128,
    'hidden_dim': 256,
    'embedding_dim': 128,
}


# ============================================================================
# UTILISATION
# ============================================================================

"""
Dans train_coco.py :

from config_coco import CONFIG, CONFIG_FAST, CONFIG_BEST

config = CONFIG        # Config par défaut
config = CONFIG_FAST   # Test rapide

# Ou personnaliser
config = {
    **CONFIG,
    'num_epochs': 20,
    'encoder_type': 'full',
}
"""
