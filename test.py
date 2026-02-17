"""
Script de Test - Vérification de l'Installation
================================================

Teste que tous les composants fonctionnent correctement
"""

import torch
import sys
import os

print("="*70)
print("TEST DE L'INSTALLATION")
print("="*70)

# ============================================================================
# 1. VÉRIFIER LES IMPORTS
# ============================================================================

print("\n[1/6] Vérification des imports...")

try:
    from utils import vocabulary, preprocessing, data_loader
    print("  ✓ utils importé avec succès")
except Exception as e:
    print(f"  ✗ Erreur d'import utils: {e}")
    sys.exit(1)

try:
    from models import  encoder, decoder, caption_model
    print("  ✓ models importé avec succès")
except Exception as e:
    print(f"  ✗ Erreur d'import models: {e}")
    sys.exit(1)

# ============================================================================
# 2. TESTER LE VOCABULAIRE
# ============================================================================

print("\n[2/6] Test du vocabulaire...")

try:
    vocab = vocabulary.Vocabulary(freq_threshold=2)
    test_captions = [
        "a dog running in the park",
        "two cats sitting on a wall",
        "the dog is playing with a ball"
    ]
    vocab.build_vocabulary(test_captions)
    
    # Test conversion
    test_text = "a dog is playing"
    indices = vocab.numericalize(test_text)
    reconstructed = vocab.denumericalize(indices)
    
    print(f"  Original:      '{test_text}'")
    print(f"  Reconstructed: '{reconstructed}'")
    print(f"  Vocab size:    {len(vocab)}")
    print("  ✓ Vocabulaire fonctionne")
except Exception as e:
    print(f"  ✗ Erreur vocabulaire: {e}")
    sys.exit(1)

# ============================================================================
# 3. TESTER L'ENCODER
# ============================================================================

print("\n[3/6] Test de l'encoder...")

try:
    encoder_lite = encoder.EncoderCNNLite(feature_dim=512)
    encoder_full = encoder.EncoderCNN(feature_dim=512)
    
    test_images = torch.randn(2, 3, 224, 224)
    
    features_lite = encoder_lite(test_images)
    features_full = encoder_full(test_images)
    
    print(f"  EncoderCNNLite:")
    print(f"    - Output shape: {features_lite.shape}")
    print(f"    - Paramètres:   {encoder_lite.get_num_params():,}")
    print(f"  EncoderCNN:")
    print(f"    - Output shape: {features_full.shape}")
    print(f"    - Paramètres:   {encoder_full.get_num_params():,}")
    print("  ✓ Encoders fonctionnent")
except Exception as e:
    print(f"  ✗ Erreur encoder: {e}")
    sys.exit(1)

# ============================================================================
# 4. TESTER LE DECODER
# ============================================================================

print("\n[4/6] Test du decoder...")

try:
    decoder_net = decoder.DecoderLSTM(
        feature_dim=512,
        embedding_dim=256,
        hidden_dim=512,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.5
    )
    
    test_features = torch.randn(2, 512)
    test_captions = torch.randint(0, len(vocab), (2, 10))
    
    outputs = decoder_net(test_features, test_captions)
    generated = decoder_net.generate(test_features[:1], max_length=15)
    
    print(f"  Training mode:")
    print(f"    - Input:  features {test_features.shape}, captions {test_captions.shape}")
    print(f"    - Output: {outputs.shape}")
    print(f"  Generation mode:")
    print(f"    - Generated: {generated.shape}")
    print(f"  Paramètres: {decoder_net.get_num_params():,}")
    print("  ✓ Decoder fonctionne")
except Exception as e:
    print(f"  ✗ Erreur decoder: {e}")
    sys.exit(1)

# ============================================================================
# 5. TESTER LE MODÈLE COMPLET
# ============================================================================

print("\n[5/6] Test du modèle complet...")

try:
    model = caption_model.create_model(
        vocab_size=len(vocab),
        embedding_dim=256,
        hidden_dim=512,
        feature_dim=512,
        num_layers=1,
        dropout=0.5,
        encoder_type='lite'
    )
    
    test_images = torch.randn(2, 3, 224, 224)
    test_captions = torch.randint(0, len(vocab), (2, 10))
    
    # Forward pass
    outputs = model(test_images, test_captions)
    
    # Generation
    caption_indices = model.generate_caption(test_images[:1])
    caption_text = vocab.denumericalize(caption_indices[0])
    
    params = model.get_num_params()
    
    print(f"  Forward pass:")
    print(f"    - Input:  images {test_images.shape}, captions {test_captions.shape}")
    print(f"    - Output: {outputs.shape}")
    print(f"  Generation:")
    print(f"    - Caption indices: {caption_indices.shape}")
    print(f"    - Caption text:    '{caption_text}'")
    print(f"  Paramètres totaux: {params['total']:,}")
    print(f"    - Encoder: {params['encoder']:,}")
    print(f"    - Decoder: {params['decoder']:,}")
    print("  ✓ Modèle complet fonctionne")
except Exception as e:
    print(f"  ✗ Erreur modèle: {e}")
    sys.exit(1)

# ============================================================================
# 6. TESTER LA LOSS
# ============================================================================

print("\n[6/6] Test du calcul de loss...")

try:
    import torch.nn as nn
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Simuler un batch
    batch_size = 2
    seq_len = 10
    vocab_size = len(vocab)
    
    outputs = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Reshape pour la loss
    outputs_reshaped = outputs.reshape(-1, vocab_size)
    targets_reshaped = targets.reshape(-1)
    
    loss = criterion(outputs_reshaped, targets_reshaped)
    
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Shape check:")
    print(f"    - Outputs: {outputs.shape} → {outputs_reshaped.shape}")
    print(f"    - Targets: {targets.shape} → {targets_reshaped.shape}")
    print("  ✓ Calcul de loss fonctionne")
except Exception as e:
    print(f"  ✗ Erreur loss: {e}")
    sys.exit(1)

# ============================================================================
# RÉSUMÉ
# ============================================================================

print("\n" + "="*70)
print("RÉSULTAT DES TESTS")
print("="*70)

print("\n✓ Tous les tests sont passés avec succès !")
print("\nVous pouvez maintenant:")
print("  1. Préparer vos données:    python prepare_data.py")
print("  2. Entraîner le modèle:     python train.py")
print("  3. Évaluer le modèle:       python evaluate.py")
print("  4. Faire une démo:          python demo.py --image <image.jpg>")

print("\n" + "="*70)

# Informations système
print("\nInformations système:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device:     {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version:    {torch.version.cuda}")
print(f"  Python version:  {sys.version.split()[0]}")

print("\n" + "="*70)