"""
Modèle complet d'Image Captioning
===================================

Quatre combinaisons encoder/decoder disponibles via encoder_type :
  'lite'      → EncoderCNNLite   + DecoderLSTM            (développement rapide)
  'full'      → EncoderCNN       + DecoderLSTM            (résiduel from scratch)
  'attention' → EncoderSpatial   + DecoderWithAttention   (résiduel + Bahdanau)
  'densenet'  → EncoderDenseNet  + DecoderWithAttention   (DenseNet-121 + Bahdanau)
                                                           ← recommandé sur COCO

Nouveautés v4 :
  - encoder_type='densenet' : EncoderDenseNet (DenseNet-121 from scratch)
    branché sur DecoderWithAttention. Même API que 'attention' pour le Trainer,
    la démo et la visualisation — aucun autre fichier à modifier.
  - Paramètres DenseNet exposés dans create_model() : growth_rate, compression,
    dense_dropout, block_config.
  - forward_with_alphas() disponible pour 'attention' et 'densenet'.
  - save/load préserve les hyper-paramètres DenseNet dans le checkpoint.
"""

import torch
import torch.nn as nn

from .encoder2 import EncoderCNNLite, EncoderCNN, EncoderSpatial, EncoderDenseNet
from .decoder2 import DecoderLSTM, DecoderWithAttention


# ============================================================================
# MODÈLE COMPLET
# ============================================================================

class ImageCaptioningModel(nn.Module):
    """Modèle complet encoder + decoder."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    def forward_with_alphas(self, images, captions):
        """
        Passe forward qui retourne également les poids d'attention (B, T, P).

        Utilisé par le Trainer pour la régularisation doubly stochastic :
            pénalité = λ · mean((1 - Σ_t alpha[t, p])²)

        Disponible avec encoder_type='attention' et encoder_type='densenet'.

        Args:
            images   : (B, 3, H, W)
            captions : (B, T)

        Returns:
            outputs : (B, T, vocab_size)
            alphas  : (B, T, num_pixels)
        """
        if not hasattr(self.decoder, 'forward_with_alphas'):
            raise ValueError(
                "forward_with_alphas nécessite encoder_type='attention' "
                "ou encoder_type='densenet'."
            )
        features = self.encoder(images)
        return self.decoder.forward_with_alphas(features, captions)

    def generate_caption(self, image, max_length=20,
                         start_token=1, end_token=2, method='greedy'):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)
            if method == 'greedy':
                return self.decoder.generate(
                    features, max_length=max_length,
                    start_token=start_token, end_token=end_token
                )
            elif method == 'beam_search':
                return self.decoder.generate_beam_search(
                    features, beam_width=3, max_length=max_length,
                    start_token=start_token, end_token=end_token
                )
            else:
                raise ValueError(f"Méthode inconnue : {method}")

    def generate_caption_with_attention(self, image, max_length=20,
                                        start_token=1, end_token=2,
                                        method='beam_search'):
        """
        Génère une caption ET retourne les poids d'attention à chaque pas.
        Disponible avec encoder_type='attention' et encoder_type='densenet'.
        """
        if not hasattr(self.decoder, 'generate_with_attention'):
            raise ValueError(
                "generate_caption_with_attention nécessite "
                "encoder_type='attention' ou encoder_type='densenet'."
            )
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)
            if method == 'greedy':
                return self.decoder.generate_with_attention(
                    features, max_length=max_length,
                    start_token=start_token, end_token=end_token
                )
            elif method == 'beam_search':
                return self.decoder.generate_beam_search_with_attention(
                    features, max_length=max_length,
                    start_token=start_token, end_token=end_token
                )
            else:
                raise ValueError(f"Méthode inconnue : {method}")

    def get_num_params(self):
        enc = self.encoder.get_num_params()
        dec = self.decoder.get_num_params()
        return {'encoder': enc, 'decoder': dec, 'total': enc + dec}


# ============================================================================
# FACTORY
# ============================================================================

def create_model(vocab_size, embedding_dim=256, hidden_dim=512, feature_dim=512,
                 num_layers=1, dropout=0.5, encoder_type='densenet',
                 attention_dim=256,
                 growth_rate=32, compression=0.5,
                 dense_dropout=0.0,
                 block_config=(6, 12, 24, 16)):
    """
    Crée le modèle complet selon encoder_type.

    encoder_type :
      'lite'      → EncoderCNNLite  + DecoderLSTM
      'full'      → EncoderCNN (résiduel) + DecoderLSTM
      'attention' → EncoderSpatial + DecoderWithAttention
      'densenet'  → EncoderDenseNet + DecoderWithAttention  ← recommandé COCO

    Paramètres DenseNet (encoder_type='densenet') :
      growth_rate  : k dans l'article — 32 (DenseNet-121)
      compression  : θ dans les transitions — 0.5
      dense_dropout: dropout dans les DenseLayers — 0.0 recommandé
      block_config : (6, 12, 24, 16) = DenseNet-121
                     (6, 12, 32, 32) = DenseNet-169 (plus lourd)

    Note : num_layers est ignoré avec 'attention' et 'densenet' (LSTMCell).
    """

    if encoder_type == 'lite':
        encoder = EncoderCNNLite(feature_dim=feature_dim)
        decoder = DecoderLSTM(
            feature_dim=feature_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            num_layers=num_layers, dropout=dropout,
        )

    elif encoder_type == 'full':
        encoder = EncoderCNN(feature_dim=feature_dim)
        decoder = DecoderLSTM(
            feature_dim=feature_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            num_layers=num_layers, dropout=dropout,
        )

    elif encoder_type == 'attention':
        encoder = EncoderSpatial(feature_dim=feature_dim, grid_size=7)
        decoder = DecoderWithAttention(
            feature_dim=feature_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            attention_dim=attention_dim, dropout=dropout,
        )

    elif encoder_type == 'densenet':
        encoder = EncoderDenseNet(
            feature_dim=feature_dim,
            grid_size=7,
            growth_rate=growth_rate,
            compression=compression,
            dropout=dense_dropout,
            block_config=block_config,
        )
        # Stocker les hyper-params DenseNet sur l'encoder pour save_model
        encoder.growth_rate  = growth_rate
        encoder.compression  = compression
        encoder.block_config = list(block_config)

        decoder = DecoderWithAttention(
            feature_dim=feature_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            attention_dim=attention_dim, dropout=dropout,
        )

    else:
        raise ValueError(
            f"encoder_type inconnu : '{encoder_type}'. "
            "Valeurs acceptées : 'lite', 'full', 'attention', 'densenet'."
        )

    return ImageCaptioningModel(encoder, decoder)


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_model(model, filepath, optimizer=None, epoch=None, loss=None, vocab=None):
    checkpoint = {
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'model_config': {
            'feature_dim':    model.encoder.feature_dim,
            'embedding_dim':  model.decoder.embedding_dim,
            'hidden_dim':     model.decoder.hidden_dim,
            'vocab_size':     model.decoder.vocab_size,
            'num_layers':     getattr(model.decoder, 'num_layers', 1),
            'encoder_type':   _detect_encoder_type(model.encoder),
            # Paramètres DenseNet — None pour les autres encoder_type
            'growth_rate':    getattr(model.encoder, 'growth_rate',  None),
            'compression':    getattr(model.encoder, 'compression',  None),
            'block_config':   getattr(model.encoder, 'block_config', None),
        }
    }

    if optimizer is not None: checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch     is not None: checkpoint['epoch'] = epoch
    if loss      is not None: checkpoint['loss']  = loss
    if vocab     is not None: checkpoint['vocab'] = vocab

    torch.save(checkpoint, filepath)
    print(f"Modèle sauvegardé dans {filepath}")


def load_model(filepath, device='cpu', encoder_type=None):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config     = checkpoint['model_config']

    etype = encoder_type or config.get('encoder_type', 'full')

    kwargs = {}
    if etype == 'densenet':
        if config.get('growth_rate')  is not None:
            kwargs['growth_rate']  = config['growth_rate']
        if config.get('compression')  is not None:
            kwargs['compression']  = config['compression']
        if config.get('block_config') is not None:
            kwargs['block_config'] = tuple(config['block_config'])

    model = create_model(
        vocab_size    = config['vocab_size'],
        embedding_dim = config['embedding_dim'],
        hidden_dim    = config['hidden_dim'],
        feature_dim   = config['feature_dim'],
        num_layers    = config.get('num_layers', 1),
        encoder_type  = etype,
        **kwargs,
    )

    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.to(device)

    info = {
        'epoch': checkpoint.get('epoch'),
        'loss':  checkpoint.get('loss'),
        'vocab': checkpoint.get('vocab'),
    }

    print(f"Modèle chargé depuis {filepath}")
    if info['epoch'] is not None: print(f"  Epoch        : {info['epoch']}")
    if info['loss']  is not None: print(f"  Loss         : {info['loss']:.4f}")
    print(f"  Encoder type : {etype}")

    return model, info


def _detect_encoder_type(encoder):
    if isinstance(encoder, EncoderDenseNet): return 'densenet'
    if isinstance(encoder, EncoderSpatial):  return 'attention'
    if isinstance(encoder, EncoderCNN):      return 'full'
    if isinstance(encoder, EncoderCNNLite):  return 'lite'
    return 'full'


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODÈLE COMPLET")
    print("="*70)

    vocab_size = 5000
    B = 2
    images = torch.randn(B, 3, 224, 224)
    caps   = torch.randint(0, vocab_size, (B, 12))

    for etype in ['lite', 'full', 'attention', 'densenet']:
        print(f"\n[encoder_type='{etype}']")
        model  = create_model(vocab_size=vocab_size, encoder_type=etype)
        out    = model(images, caps)
        params = model.get_num_params()
        print(f"  Forward : {out.shape}")
        print(f"  Params  : encoder={params['encoder']:,}  "
              f"decoder={params['decoder']:,}  total={params['total']:,}")

        if etype in ('attention', 'densenet'):
            out_a, alphas = model.forward_with_alphas(images, caps)
            print(f"  Alphas  : {alphas.shape}")

    print("\n" + "="*70)
    print("Tous les modèles fonctionnent !")
    print("="*70)