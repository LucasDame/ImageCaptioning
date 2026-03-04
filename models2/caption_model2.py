"""
Modèle complet d'Image Captioning
===================================

Améliorations par rapport à l'original :

1. Trois combinaisons encoder/decoder disponibles via encoder_type :
     'lite'      → EncoderCNNLite  + DecoderLSTM           (développement rapide)
     'full'      → EncoderCNN      + DecoderLSTM           (résiduel from scratch)
     'attention' → EncoderSpatial  + DecoderWithAttention  (meilleure qualité)

2. save_model sauvegarde le encoder_type dans le checkpoint pour éviter
   de devoir le re-spécifier au chargement.

Compatibilité ascendante :
   encoder_type='lite' et encoder_type='full' fonctionnent exactement comme
   dans l'original (même API dans train.py, evaluate.py, config.py).
"""

import torch
import torch.nn as nn

from .encoder2 import EncoderCNNLite, EncoderCNN, EncoderSpatial
from .decoder2 import DecoderLSTM, DecoderWithAttention


# ============================================================================
# MODÈLE COMPLET
# ============================================================================

class ImageCaptioningModel(nn.Module):
    """
    Modèle complet encoder + decoder. Inchangé par rapport à l'original.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

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

    def get_num_params(self):
        enc = self.encoder.get_num_params()
        dec = self.decoder.get_num_params()
        return {'encoder': enc, 'decoder': dec, 'total': enc + dec}


# ============================================================================
# FACTORY
# ============================================================================

def create_model(vocab_size, embedding_dim=256, hidden_dim=512, feature_dim=512,
                 num_layers=1, dropout=0.5, encoder_type='full',
                 attention_dim=256):
    """
    Crée le modèle complet selon encoder_type.

    encoder_type :
      'lite'      → EncoderCNNLite  + DecoderLSTM
                    Développement rapide, ~2 M params
      'full'      → EncoderCNN (résiduel) + DecoderLSTM
                    CNN from scratch avec blocs résiduels, ~15 M params
      'attention' → EncoderSpatial + DecoderWithAttention
                    Résiduel + attention de Bahdanau, meilleure qualité

    Args:
        vocab_size    : taille du vocabulaire
        embedding_dim : dimension des word embeddings
        hidden_dim    : hidden state du LSTM
        feature_dim   : dimension des features de l'encoder
        num_layers    : couches LSTM (DecoderLSTM uniquement)
        dropout       : dropout
        encoder_type  : 'lite', 'full' ou 'attention'
        attention_dim : dimension interne de l'attention (mode 'attention')
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

    else:
        raise ValueError(
            f"encoder_type inconnu : '{encoder_type}'. "
            "Valeurs acceptées : 'lite', 'full', 'attention'."
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

    model = create_model(
        vocab_size    = config['vocab_size'],
        embedding_dim = config['embedding_dim'],
        hidden_dim    = config['hidden_dim'],
        feature_dim   = config['feature_dim'],
        num_layers    = config.get('num_layers', 1),
        encoder_type  = etype,
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
    if info['epoch'] is not None: print(f"  Epoch : {info['epoch']}")
    if info['loss']  is not None: print(f"  Loss  : {info['loss']:.4f}")

    return model, info


def _detect_encoder_type(encoder):
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

    for etype in ['lite', 'full', 'attention']:
        print(f"\n[encoder_type='{etype}']")
        model  = create_model(vocab_size=vocab_size, encoder_type=etype)
        out    = model(images, caps)
        params = model.get_num_params()
        print(f"  Forward : {out.shape}")
        print(f"  Params  : encoder={params['encoder']:,}  "
              f"decoder={params['decoder']:,}  total={params['total']:,}")

        gen = model.generate_caption(images[:1], max_length=8, method='beam_search')
        print(f"  Beam search : {gen.shape} → {gen[0].tolist()}")

    print("\n" + "="*70)
    print("Tous les modèles fonctionnent !")
    print("="*70)
