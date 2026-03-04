"""
Package `models` pour ImageCaptioning.

Expose les sous-modules `encoder`, `decoder` et `caption_model` via des imports relatifs
pour éviter les collisions avec des modules top-level nommés identiquement.
"""

from . import caption_model2, decoder2, encoder2

__all__ = ["encoder2", "decoder2", "caption_model2"]
