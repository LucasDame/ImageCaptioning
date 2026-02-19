"""
Modèle complet d'Image Captioning
==================================

Combine l'encoder CNN et le decoder LSTM.
"""

import torch
import torch.nn as nn
from .encoder import EncoderCNN, EncoderCNNLite
from .decoder import DecoderLSTM


class ImageCaptioningModel(nn.Module):
    """
    Modèle complet pour l'image captioning
    Combine un encoder CNN et un decoder LSTM
    """
    
    def __init__(self, encoder, decoder):
        """
        Args:
            encoder (nn.Module): Encoder CNN (EncoderCNN ou EncoderCNNLite)
            decoder (nn.Module): Decoder LSTM (DecoderLSTM)
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, images, captions):
        """
        Forward pass complet (mode entraînement)
        
        Args:
            images (torch.Tensor): Batch d'images
                                  Shape: (batch_size, 3, 224, 224)
            captions (torch.Tensor): Batch de captions
                                    Shape: (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Prédictions
                         Shape: (batch_size, seq_len, vocab_size)
        """
        # 1. Extraire les features des images
        features = self.encoder(images)  # (batch_size, feature_dim)
        
        # 2. Générer les prédictions de mots
        outputs = self.decoder(features, captions)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
    def generate_caption(self, image, max_length=20, start_token=1, end_token=2, method='greedy'):
        """
        Génère une caption pour une seule image
        
        Args:
            image (torch.Tensor): Image unique
                                 Shape: (1, 3, 224, 224) ou (3, 224, 224)
            max_length (int): Longueur maximale de la caption
            start_token (int): Index du token <START>
            end_token (int): Index du token <END>
            method (str): 'greedy' ou 'beam_search'
        
        Returns:
            torch.Tensor: Caption générée (indices)
                         Shape: (1, seq_len)
        """
        # S'assurer que l'image a la bonne shape
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Ajouter batch dimension
        
        # Mode évaluation
        self.eval()
        
        with torch.no_grad():
            # Extraire les features
            features = self.encoder(image)  # (1, feature_dim)
            
            # Générer la caption
            if method == 'greedy':
                caption = self.decoder.generate(
                    features, 
                    max_length=max_length,
                    start_token=start_token,
                    end_token=end_token
                )
            elif method == 'beam_search':
                caption = self.decoder.generate_beam_search(
                    features,
                    beam_width=3,
                    max_length=max_length,
                    start_token=start_token,
                    end_token=end_token
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return caption
    
    def get_num_params(self):
        """
        Retourne le nombre total de paramètres
        """
        encoder_params = self.encoder.get_num_params()
        decoder_params = self.decoder.get_num_params()
        total_params = encoder_params + decoder_params
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }


def create_model(vocab_size, embedding_dim=256, hidden_dim=512, feature_dim=512, 
                num_layers=1, dropout=0.5, encoder_type='full'):
    """
    Factory function pour créer le modèle complet
    
    Args:
        vocab_size (int): Taille du vocabulaire
        embedding_dim (int): Dimension des word embeddings
        hidden_dim (int): Dimension du LSTM hidden state
        feature_dim (int): Dimension des features de l'encoder
        num_layers (int): Nombre de couches LSTM
        dropout (float): Taux de dropout
        encoder_type (str): 'full' ou 'lite'
    
    Returns:
        ImageCaptioningModel: Modèle complet
    """
    # Créer l'encoder
    if encoder_type == 'full':
        encoder = EncoderCNN(feature_dim=feature_dim)
    elif encoder_type == 'lite':
        encoder = EncoderCNNLite(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Créer le decoder
    decoder = DecoderLSTM(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Créer le modèle complet
    model = ImageCaptioningModel(encoder, decoder)
    
    return model


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def save_model(model, filepath, optimizer=None, epoch=None, loss=None, vocab=None):
    """
    Sauvegarde le modèle et les informations d'entraînement
    
    Args:
        model (ImageCaptioningModel): Modèle à sauvegarder
        filepath (str): Chemin de sauvegarde
        optimizer (torch.optim.Optimizer): Optimiseur (optionnel)
        epoch (int): Numéro de l'epoch (optionnel)
        loss (float): Loss actuelle (optionnel)
        vocab (Vocabulary): Vocabulaire (optionnel)
    """
    checkpoint = {
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'model_config': {
            'feature_dim': model.encoder.feature_dim,
            'embedding_dim': model.decoder.embedding_dim,
            'hidden_dim': model.decoder.hidden_dim,
            'vocab_size': model.decoder.vocab_size,
            'num_layers': model.decoder.num_layers
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if vocab is not None:
        checkpoint['vocab'] = vocab
    
    torch.save(checkpoint, filepath)
    print(f"Modèle sauvegardé dans {filepath}")


def load_model(filepath, device='cpu', encoder_type='full'):
    """
    Charge un modèle sauvegardé
    
    Args:
        filepath (str): Chemin du checkpoint
        device (str): 'cpu' ou 'cuda'
        encoder_type (str): Type d'encoder utilisé lors de la sauvegarde
    
    Returns:
        tuple: (model, checkpoint_info)
    """
    # Charger avec weights_only=False pour permettre le chargement du vocabulaire
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    config = checkpoint['model_config']
    
    # Recréer le modèle avec la même architecture
    model = create_model(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        feature_dim=config['feature_dim'],
        num_layers=config['num_layers'],
        encoder_type=encoder_type
    )
    
    # Charger les poids
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    model.to(device)
    
    # Informations supplémentaires
    info = {
        'epoch': checkpoint.get('epoch', None),
        'loss': checkpoint.get('loss', None),
        'vocab': checkpoint.get('vocab', None)
    }
    
    print(f"Modèle chargé depuis {filepath}")
    if info['epoch'] is not None:
        print(f"  Epoch: {info['epoch']}")
    if info['loss'] is not None:
        print(f"  Loss: {info['loss']:.4f}")
    
    return model, info


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODÈLE COMPLET")
    print("="*70)
    
    # Créer le modèle
    vocab_size = 5000
    model = create_model(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        feature_dim=512,
        num_layers=1,
        dropout=0.5,
        encoder_type='lite'  # Utiliser la version lite pour tester
    )
    
    # Afficher les paramètres
    params = model.get_num_params()
    print(f"\nNombre de paramètres:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total:   {params['total']:,}")
    
    # Test forward pass
    print("\n[TEST 1] Forward pass (entraînement)")
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 15))
    
    outputs = model(images, captions)
    print(f"  Input images shape: {images.shape}")
    print(f"  Input captions shape: {captions.shape}")
    print(f"  Output shape: {outputs.shape}")
    
    # Test caption generation
    print("\n[TEST 2] Caption generation (inférence)")
    test_image = torch.randn(1, 3, 224, 224)
    caption = model.generate_caption(test_image, max_length=20)
    print(f"  Input image shape: {test_image.shape}")
    print(f"  Generated caption shape: {caption.shape}")
    print(f"  Generated indices: {caption[0].tolist()[:10]}...")
    
    # Test save/load
    print("\n[TEST 3] Sauvegarde et chargement")
    save_model(model, 'test_checkpoint.pth', epoch=0, loss=1.5)
    loaded_model, info = load_model('test_checkpoint.pth', encoder_type='lite')
    print(f"  Modèle rechargé avec succès")
    
    import os
    os.remove('test_checkpoint.pth')
    print(f"  Fichier de test supprimé")
    
    print("\n" + "="*70)
    print("Modèle complet fonctionne correctement !")
    print("="*70)