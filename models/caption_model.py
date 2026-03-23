"""
caption_model.py — Modèle complet d'Image Captioning
======================================================

Trois architectures disponibles via le paramètre `model` :
  'cnn'       → EncoderCNN       + DecoderLSTM            (résiduel from scratch, vecteur global)
  'resnet'    → EncoderSpatial   + DecoderWithAttention   (résiduel from scratch + Bahdanau)
  'densenet'  → EncoderDenseNet  + DecoderWithAttention   (DenseNet-121 from scratch + Bahdanau)

Notes :
  - 'cnn' produit un vecteur global → DecoderLSTM standard (pas d'attention).
  - 'resnet' et 'densenet' produisent une grille 7×7 → DecoderWithAttention.
  - forward_with_alphas() et generate_caption_with_attention() ne sont
    disponibles qu'avec 'resnet' et 'densenet'.
"""

import os
import torch
import torch.nn as nn

from .encoder import EncoderCNN, EncoderSpatial, EncoderDenseNet
from .decoder import DecoderLSTM, DecoderWithAttention


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
        Passe forward retournant aussi les poids d'attention (B, T, P).
        Utilisé pour la régularisation doubly stochastic (Xu et al. 2015).
        Disponible uniquement avec model='resnet' ou model='densenet'.
        """
        if not hasattr(self.decoder, 'forward_with_alphas'):
            raise ValueError(
                "forward_with_alphas nécessite model='resnet' ou model='densenet'."
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
        Génère une caption ET retourne les poids d'attention.
        Disponible uniquement avec model='resnet' ou model='densenet'.
        """
        if not hasattr(self.decoder, 'generate_with_attention'):
            raise ValueError(
                "generate_caption_with_attention nécessite "
                "model='resnet' ou model='densenet'."
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

    def generate_diverse_captions(self, image, num_captions=5,
                                  beam_width=5, max_length=20,
                                  start_token=1, end_token=2,
                                  diversity_penalty=0.8):
        """
        Génère num_captions captions différentes et pertinentes via Diverse Beam Search.

        Disponible pour tous les modèles (cnn, resnet, densenet).

        Args:
            image            : (3, H, W) ou (1, 3, H, W)
            num_captions     : nombre de captions à générer (défaut: 5)
            beam_width       : taille du faisceau par groupe (défaut: 5)
            max_length       : longueur max de chaque caption
            start_token      : index START (défaut: 1)
            end_token        : index END   (défaut: 2)
            diversity_penalty: force de la diversité, 0.8 recommandé
                               (0 = beam search standard, 2 = très diversifié)

        Returns:
            list[Tensor] : num_captions tensors (1, seq_len), du meilleur au moins bon
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)
            return self.decoder.generate_diverse_beam_search(
                features,
                num_captions=num_captions,
                beam_width=beam_width,
                max_length=max_length,
                start_token=start_token,
                end_token=end_token,
                diversity_penalty=diversity_penalty,
            )

    def get_num_params(self):
        enc = self.encoder.get_num_params()
        dec = self.decoder.get_num_params()
        return {'encoder': enc, 'decoder': dec, 'total': enc + dec}


# ============================================================================
# FACTORY
# ============================================================================

def create_model(vocab_size, embedding_dim=256, hidden_dim=512, feature_dim=512,
                 dropout=0.5, model='densenet', attention_dim=256,
                 growth_rate=32, compression=0.5, dense_dropout=0.0,
                 block_config=(6, 12, 24, 16)):
    """
    Crée le modèle complet selon l'architecture choisie.

    Args:
        vocab_size    : taille du vocabulaire
        embedding_dim : dimension des embeddings de mots
        hidden_dim    : dimension cachée du LSTM
        feature_dim   : dimension des features de l'encodeur
        dropout       : taux de dropout
        model         : 'cnn', 'resnet' ou 'densenet'
        attention_dim : dimension de l'espace d'attention (resnet / densenet)
        growth_rate   : k DenseNet — 32 pour DenseNet-121 (densenet uniquement)
        compression   : θ DenseNet — 0.5 standard          (densenet uniquement)
        dense_dropout : dropout dans les DenseLayers        (densenet uniquement)
        block_config  : (6,12,24,16)=DenseNet-121          (densenet uniquement)

    Returns:
        ImageCaptioningModel
    """
    if model == 'cnn':
        encoder = EncoderCNN(feature_dim=feature_dim)
        decoder = DecoderLSTM(
            feature_dim=feature_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            num_layers=1, dropout=dropout,
        )

    elif model == 'resnet':
        encoder = EncoderSpatial(feature_dim=feature_dim, grid_size=7)
        decoder = DecoderWithAttention(
            feature_dim=feature_dim, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            attention_dim=attention_dim, dropout=dropout,
        )

    elif model == 'densenet':
        encoder = EncoderDenseNet(
            feature_dim=feature_dim, grid_size=7,
            growth_rate=growth_rate, compression=compression,
            dropout=dense_dropout, block_config=block_config,
        )
        # Stocker les hyper-params DenseNet pour save_model
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
            f"Architecture inconnue : '{model}'. "
            "Valeurs acceptées : 'cnn', 'resnet', 'densenet'."
        )

    return ImageCaptioningModel(encoder, decoder)


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_model(model, filepath, optimizer=None, epoch=None, loss=None,
               vocab=None, scheduler_state=None):
    """
    Sauvegarde le modèle avec toute la configuration nécessaire pour le recharger.
    """
    checkpoint = {
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'model_config': {
            'feature_dim':    model.encoder.feature_dim,
            'embedding_dim':  model.decoder.embedding_dim,
            'hidden_dim':     model.decoder.hidden_dim,
            'vocab_size':     model.decoder.vocab_size,
            'model':          _detect_model_type(model.encoder),
            # attention_dim : présent uniquement sur DecoderWithAttention
            'attention_dim':  getattr(model.decoder, 'attention_dim', None),
            # Paramètres DenseNet (None pour les autres architectures)
            'growth_rate':    getattr(model.encoder, 'growth_rate',  None),
            'compression':    getattr(model.encoder, 'compression',  None),
            'block_config':   getattr(model.encoder, 'block_config', None),
        }
    }

    if optimizer       is not None: checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler_state is not None: checkpoint['scheduler_state_dict'] = scheduler_state
    if epoch           is not None: checkpoint['epoch'] = epoch
    if loss            is not None: checkpoint['loss']  = loss
    if vocab           is not None: checkpoint['vocab'] = vocab

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Modèle sauvegardé → {filepath}")


def load_model(filepath, device='cpu', model=None):
    """
    Charge un modèle depuis un checkpoint.
    L'architecture est détectée automatiquement depuis le checkpoint.

    Returns:
        tuple (ImageCaptioningModel, info_dict)
        info_dict : epoch, loss, vocab, scheduler_state
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config     = checkpoint['model_config']

    arch = model or config.get('model', 'densenet')

    kwargs = {}
    # Restaurer attention_dim pour les architectures avec attention
    if arch in ('resnet', 'densenet') and config.get('attention_dim') is not None:
        kwargs['attention_dim'] = config['attention_dim']
    # Restaurer les hyperparamètres DenseNet
    if arch == 'densenet':
        if config.get('growth_rate')  is not None:
            kwargs['growth_rate']  = config['growth_rate']
        if config.get('compression')  is not None:
            kwargs['compression']  = config['compression']
        if config.get('block_config') is not None:
            kwargs['block_config'] = tuple(config['block_config'])

    m = create_model(
        vocab_size    = config['vocab_size'],
        embedding_dim = config['embedding_dim'],
        hidden_dim    = config['hidden_dim'],
        feature_dim   = config['feature_dim'],
        model         = arch,
        **kwargs,
    )

    m.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    m.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    m.to(device)

    info = {
        'epoch':           checkpoint.get('epoch'),
        'loss':            checkpoint.get('loss'),
        'vocab':           checkpoint.get('vocab'),
        'scheduler_state': checkpoint.get('scheduler_state_dict'),
    }

    print(f"Modèle chargé depuis {filepath}")
    if info['epoch'] is not None: print(f"  Epoch        : {info['epoch']}")
    if info['loss']  is not None: print(f"  Loss         : {info['loss']:.4f}")
    print(f"  Architecture : {arch}")

    return m, info


def _detect_model_type(encoder):
    if isinstance(encoder, EncoderDenseNet): return 'densenet'
    if isinstance(encoder, EncoderSpatial):  return 'resnet'
    if isinstance(encoder, EncoderCNN):      return 'cnn'
    return 'cnn'


# ============================================================================
# TESTS RAPIDES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODÈLE COMPLET")
    print("="*70)

    vocab_size = 5000
    B = 2
    images = torch.randn(B, 3, 224, 224)
    caps   = torch.randint(0, vocab_size, (B, 12))

    for arch in ['cnn', 'resnet', 'densenet']:
        print(f"\n[model='{arch}']")
        m   = create_model(vocab_size=vocab_size, model=arch)
        out = m(images, caps)
        p   = m.get_num_params()
        print(f"  Forward : {out.shape}")
        print(f"  Params  : encoder={p['encoder']:,}  decoder={p['decoder']:,}  total={p['total']:,}")
        if arch in ('resnet', 'densenet'):
            out_a, alphas = m.forward_with_alphas(images, caps)
            print(f"  Alphas  : {alphas.shape}")

    print("\n" + "="*70)
    print("Tous les modèles fonctionnent !")
    print("="*70)