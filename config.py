"""
Configuration Template
======================

Copiez ce fichier et modifiez les valeurs selon vos besoins.
"""

CONFIG = {
    # ========================================================================
    # CHEMINS
    # ========================================================================
    'data_dir': 'data',
    'captions_file': 'data/flicker8k/captions.txt',
    'images_dir': 'data/flicker8k/Images',
    'vocab_path': 'data/vocab.pkl',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'results_dir': 'results',
    
    # ========================================================================
    # HYPERPARAMÈTRES DU MODÈLE
    # ========================================================================
    
    # Dimensions
    'embedding_dim': 256,        # Dimension des word embeddings
                                # Recommandé: 128-512
                                # Plus grand = plus de capacité mais plus lent
    
    'hidden_dim': 512,          # Dimension du LSTM hidden state
                                # Recommandé: 256-1024
                                # Plus grand = plus de capacité mais plus lent
    
    'feature_dim': 512,         # Dimension des features de l'encoder
                                # Doit correspondre à la sortie de l'encoder
                                # Standard: 512
    
    'num_layers': 1,            # Nombre de couches LSTM
                                # 1: Rapide, simple
                                # 2-3: Plus de capacité mais risque d'overfitting
    
    'dropout': 0.5,             # Taux de dropout (régularisation)
                                # 0.3: Léger
                                # 0.5: Standard
                                # 0.7: Fort (si beaucoup d'overfitting)
    
    'encoder_type': 'lite',     # Type d'encoder: 'lite' ou 'full'
                                # 'lite': ~2M params, rapide, bon pour développer
                                # 'full': ~15M params, meilleure qualité
    
    # ========================================================================
    # HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # ========================================================================
    
    'num_epochs': 50,           # Nombre d'epochs
                                # Test rapide: 5
                                # Standard: 30
                                # Meilleur résultat: 50+

    'patience': 5,             # Patience pour l'early stopping
    
    'batch_size': 32,           # Taille du batch
                                # Plus grand = plus rapide mais plus de mémoire
                                # 8: Si peu de mémoire GPU
                                # 16: Standard pour petites GPU
                                # 32: Standard pour GPU moyennes
                                # 64: Pour grandes GPU
    
    'learning_rate': 0.001,     # Learning rate (Adam)
                                # 0.0001: Très prudent
                                # 0.001: Standard (recommandé)
                                # 0.01: Agressif (risque d'instabilité)
    
    'weight_decay': 1e-5,       # Régularisation L2
                                # Standard: 1e-5
                                # Plus de régularisation: 1e-4
    
    'num_workers': 4,           # Workers pour le DataLoader
                                # 0: Pas de parallélisation
                                # 2-4: Standard
                                # 8+: Si beaucoup de CPU
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    
    'image_size': 224,          # Taille des images (carré)
                                # Standard: 224x224
                                # Plus rapide: 128x128
                                # Plus précis: 299x299
    
    'freq_threshold': 5,        # Fréquence minimale pour inclure un mot
                                # Trop bas (1-2): Vocabulaire très grand
                                # Standard: 5
                                # Trop haut (10+): Beaucoup de <UNK>
    
    'train_ratio': 0.8,         # Ratio de données d'entraînement
    'val_ratio': 0.1,           # Ratio de validation (rest = test)
    'random_seed': 42,          # Seed pour la reproductibilité
    
    # ========================================================================
    # SAUVEGARDE
    # ========================================================================
    
    'save_every': 5,            # Sauvegarder un checkpoint tous les N epochs
    
    # ========================================================================
    # GÉNÉRATION (DÉMO/ÉVALUATION)
    # ========================================================================
    
    'max_caption_length': 20,   # Longueur maximale des captions générées
    'generation_method': 'greedy',  # 'greedy' ou 'beam_search'
}


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# ============================================================================

# Configuration pour un test rapide (développement)
CONFIG_FAST = {
    **CONFIG,
    'num_epochs': 5,
    'encoder_type': 'lite',
    'batch_size': 64,
    'embedding_dim': 128,
    'hidden_dim': 256,
}

# Configuration pour la qualité maximale (entraînement final)
CONFIG_BEST = {
    **CONFIG,
    'num_epochs': 50,
    'encoder_type': 'full',
    'batch_size': 32,
    'embedding_dim': 512,
    'hidden_dim': 1024,
    'num_layers': 2,
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
Dans train.py, vous pouvez faire:

from config import CONFIG, CONFIG_FAST, CONFIG_BEST

# Utiliser la config par défaut
config = CONFIG

# Ou une config prédéfinie
config = CONFIG_FAST  # Pour tester rapidement

# Ou personnaliser
config = {
    **CONFIG,
    'num_epochs': 40,
    'encoder_type': 'full',
}
"""