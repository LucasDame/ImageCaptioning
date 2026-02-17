"""
Package `models` pour ImageCaptioning.

Expose les sous-modules `encoder`, `decoder` et `caption_model` via des imports relatifs
pour éviter les collisions avec des modules top-level nommés identiquement.
"""

from . import encoder, decoder, caption_model

__all__ = ["encoder", "decoder", "caption_model"]
