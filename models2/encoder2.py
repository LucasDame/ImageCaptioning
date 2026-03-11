"""
Encoder CNN pour Image Captioning
==================================

Architectures disponibles :
  - EncoderCNNLite     : version légère pour le développement rapide (inchangée)
  - EncoderCNN         : CNN from scratch avec blocs résiduels
  - EncoderSpatial     : résiduel from scratch, retourne une grille 7×7
                         (à utiliser avec DecoderWithAttention)
  - EncoderDenseNet    : DenseNet-121 from scratch, retourne une grille 7×7
                         (meilleure option pour l'attention sur COCO)

Pourquoi DenseNet pour le captioning ?
  Dans un ResNet, chaque couche ne reçoit que la sortie de la couche précédente.
  Dans un DenseNet, chaque couche reçoit la concaténation de toutes les couches
  précédentes du même bloc. La grille spatiale finale agrège donc des features
  à toutes les échelles : bords et textures (couches précoces) ET formes et
  objets complexes (couches tardives). L'attention Bahdanau tire pleinement
  parti de cette richesse : pour générer "airplane" elle peut s'appuyer sur des
  features de forme haut-niveau, pour "runway" sur des features de texture
  bas-niveau — dans la même passe forward.

  Avantages mesurés sur COCO + attention LSTM :
    - Meilleur gradient flow during training from scratch (pas de pretrained)
    - Réutilisation des features → moins de paramètres pour une profondeur équivalente
    - BLEU-4 et CIDEr supérieurs aux mini-ResNets de profondeur comparable

Architecture DenseNet-121 (from scratch) :
  Stem          : Conv7×7 stride 2 → BN → ReLU → MaxPool    224 → 56
  DenseBlock1   : 6 couches, growth_rate=32                  56  → 56  (ch: 64→256)
  Transition1   : Conv1×1 (θ=0.5) → AvgPool2                56  → 28  (ch: 256→128)
  DenseBlock2   : 12 couches, growth_rate=32                 28  → 28  (ch: 128→512)
  Transition2   : Conv1×1 (θ=0.5) → AvgPool2                28  → 14  (ch: 512→256)
  DenseBlock3   : 24 couches, growth_rate=32                 14  → 14  (ch: 256→1024)
  Transition3   : Conv1×1 (θ=0.5) → AvgPool2                14  → 7   (ch: 1024→512)
  DenseBlock4   : 16 couches, growth_rate=32                 7   → 7   (ch: 512→1024)
  BN → ReLU → AdaptiveAvgPool(7,7) → projection Linear pixelwise
  Sortie        : (B, 49, feature_dim)

Aucun modèle pré-entraîné n'est utilisé.
"""

import torch
import torch.nn as nn


# ============================================================================
# BLOC RÉSIDUEL (pour EncoderCNN / EncoderSpatial — inchangé)
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Bloc Conv→BN→ReLU→Conv→BN avec connexion résiduelle.
    Utilisé par EncoderCNN et EncoderSpatial.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))


# ============================================================================
# BLOCS DENSENET
# ============================================================================

class _DenseLayer(nn.Module):
    """
    Une couche dans un DenseBlock.

    Architecture : BN → ReLU → Conv1×1 (bottleneck) → BN → ReLU → Conv3×3
    La couche reçoit x et retourne cat([x, out]) — la connexion dense.

    Le bottleneck 1×1 compresse les canaux d'entrée (qui grandissent à chaque
    couche) à 4*growth_rate avant la conv 3×3, ce qui limite l'explosion du
    nombre de paramètres tout en conservant l'accès à toutes les features.

    Args:
        in_channels  : canaux d'entrée (64 + i*growth_rate pour la couche i)
        growth_rate  : k dans l'article DenseNet — nombre de feature maps
                       produites par cette couche, ajouté aux canaux d'entrée
        dropout      : dropout appliqué après la conv 3×3
    """
    def __init__(self, in_channels, growth_rate, dropout=0.0):
        super().__init__()
        # Bottleneck : réduit in_channels → 4*k avant la conv 3×3
        bn_size = 4 * growth_rate
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size, kernel_size=1, bias=False),

            nn.BatchNorm2d(bn_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size, growth_rate, kernel_size=3,
                      padding=1, bias=False),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Connexion dense : concatène l'entrée avec la nouvelle sortie
        return torch.cat([x, self.drop(self.layer(x))], dim=1)


class _DenseBlock(nn.Module):
    """
    Bloc de num_layers couches denses empilées.

    Après num_layers couches, les canaux de sortie valent :
        out_channels = in_channels + num_layers * growth_rate

    Args:
        in_channels : canaux en entrée du bloc
        num_layers  : nombre de DenseLayers
        growth_rate : k — canaux ajoutés par chaque couche
        dropout     : passé à chaque DenseLayer
    """
    def __init__(self, in_channels, num_layers, growth_rate, dropout=0.0):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth_rate, dropout))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class _TransitionLayer(nn.Module):
    """
    Couche de transition entre deux DenseBlocks.

    BN → ReLU → Conv1×1 (compression θ) → AvgPool2×2

    La compression θ=0.5 réduit le nombre de canaux de moitié avant le
    downsampling, ce qui limite la mémoire et force une compaction des features.

    Args:
        in_channels  : canaux en entrée
        compression  : θ ∈ ]0, 1] — fraction des canaux conservés
    """
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = out_channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer(x)


# ============================================================================
# ENCODER DENSENET SPATIAL (for scratch — attention visuelle)
# ============================================================================

class EncoderDenseNet(nn.Module):
    """
    DenseNet-121 from scratch adapté pour l'attention visuelle.

    Retourne une grille spatiale (grid_size × grid_size) de features au lieu
    d'un vecteur global, exactement comme EncoderSpatial — compatible avec
    DecoderWithAttention sans aucun changement dans le reste du code.

    Architecture (DenseNet-121, growth_rate=32) :
      Stem          : Conv7×7/2 → BN → ReLU → MaxPool3×3/2    224 → 56
      DenseBlock1   :  6 couches                               56  → 56   ch→256
      Transition1   : θ=0.5, AvgPool/2                        56  → 28   ch→128
      DenseBlock2   : 12 couches                               28  → 28   ch→512
      Transition2   : θ=0.5, AvgPool/2                        28  → 14   ch→256
      DenseBlock3   : 24 couches                               14  → 14   ch→1024
      Transition3   : θ=0.5, AvgPool/2                        14  → 7    ch→512
      DenseBlock4   : 16 couches                               7   → 7    ch→1024
      BN → ReLU
      AdaptiveAvgPool(grid_size, grid_size)                    7   → G×G
      Linear pixelwise (1024 → feature_dim)                        → feature_dim

    Sortie : (batch_size, grid_size², feature_dim)
             ex: (B, 49, 512) pour grid_size=7

    Pourquoi cette configuration (6-12-24-16, k=32) ?
      C'est la configuration DenseNet-121 de l'article original (Huang et al.
      2017). Elle offre le meilleur compromis profondeur/paramètres pour les
      tâches de vision from scratch, et a été validée dans la littérature sur
      COCO + attention pour le captioning.

    Args:
        feature_dim  : dimension de projection finale (= hidden_dim du decoder)
        grid_size    : taille de la grille spatiale (7 → 49 régions)
        growth_rate  : k dans l'article — 32 par défaut (DenseNet-121)
        compression  : θ dans les transitions — 0.5 par défaut
        dropout      : dropout dans les DenseLayers (0.0 = désactivé)
        block_config : tuple des num_layers par DenseBlock
                       (6, 12, 24, 16) = DenseNet-121
                       (6, 12, 32, 32) = DenseNet-169 (plus lourd)
    """

    def __init__(self, feature_dim=512, grid_size=7,
                 growth_rate=32, compression=0.5, dropout=0.0,
                 block_config=(6, 12, 24, 16)):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size   = grid_size

        # ── Stem ──────────────────────────────────────────────────────────────
        # Identique au stem ResNet / DenseNet original :
        # Conv7×7 stride 2 pour un downsampling agressif dès le début,
        # suivi d'un MaxPool pour passer de 224 à 56.
        num_init_features = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7,
                      stride=2, padding=3, bias=False),          # 224 → 112
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # 112 → 56
        )

        # ── DenseBlocks + Transitions ─────────────────────────────────────────
        # On construit dynamiquement les blocs en suivant block_config.
        # Après chaque DenseBlock sauf le dernier, une TransitionLayer
        # réduit les canaux (×θ) et la résolution spatiale (÷2).
        ch = num_init_features
        dense_layers = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(ch, num_layers, growth_rate, dropout)
            dense_layers.append(block)
            ch = block.out_channels

            # Pas de transition après le dernier bloc
            if i < len(block_config) - 1:
                trans = _TransitionLayer(ch, compression)
                dense_layers.append(trans)
                ch = trans.out_channels

        self.dense_layers = nn.Sequential(*dense_layers)
        self.final_ch     = ch   # = 1024 pour DenseNet-121 (k=32)

        # ── BN final + pool spatial ────────────────────────────────────────────
        # BN+ReLU final recommandé par l'article DenseNet original.
        # AdaptiveAvgPool garantit une sortie G×G quelle que soit la résolution
        # intermédiaire (utile si on change grid_size ou image_size).
        self.final_norm = nn.Sequential(
            nn.BatchNorm2d(self.final_ch),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # ── Projection pixelwise vers feature_dim ─────────────────────────────
        # Appliquée indépendamment sur chacune des G² régions spatiales.
        # Linear(1024 → feature_dim) sans activation ni dropout :
        # l'activation est dans le BN final, le dropout est dans les DenseLayers.
        self.fc = nn.Linear(self.final_ch, feature_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, images):
        """
        Args:
            images : (B, 3, 224, 224)

        Returns:
            (B, grid_size², feature_dim)
        """
        x = self.stem(images)            # (B,  64, 56, 56)
        x = self.dense_layers(x)         # (B, 1024,  7,  7)  ← après DB4
        x = self.final_norm(x)           # (B, 1024,  7,  7)
        x = self.adaptive_pool(x)        # (B, 1024,  G,  G)

        B, C, G, _ = x.shape
        x = x.permute(0, 2, 3, 1)       # (B, G, G, C)
        x = x.reshape(B, G * G, C)      # (B, G², C)
        return self.fc(x)               # (B, G², feature_dim)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER RÉSIDUEL (from scratch) — inchangé
# ============================================================================

class EncoderCNN(nn.Module):
    """
    CNN from scratch avec blocs résiduels.

    Architecture :
      Stem  : Conv7×7 stride 2 → BN → ReLU → MaxPool   224→56
      Layer1: ResidualBlock(64  → 128, stride 2)         56→28
      Layer2: ResidualBlock(128 → 256, stride 2)         28→14
      Layer3: ResidualBlock(256 → 512, stride 2)         14→7
      Layer4: ResidualBlock(512 → 512, stride 1)          7→7
      AdaptiveAvgPool → flatten → Linear → ReLU → Dropout

    Sortie : (batch_size, feature_dim)
    """

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = ResidualBlock(64,  128, stride=2)
        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.layer3 = ResidualBlock(256, 512, stride=2)
        self.layer4 = ResidualBlock(512, 512, stride=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER SPATIAL RÉSIDUEL (pour l'attention — from scratch) — inchangé
# ============================================================================

class EncoderSpatial(nn.Module):
    """
    Même architecture résiduelle que EncoderCNN, mais retourne une carte
    spatiale (grid_size × grid_size) au lieu d'un vecteur global.

    Utilisé avec DecoderWithAttention.
    Sortie : (batch_size, grid_size², feature_dim)
    """

    def __init__(self, feature_dim=512, grid_size=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size   = grid_size

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = ResidualBlock(64,  128, stride=2)
        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.layer3 = ResidualBlock(256, 512, stride=2)
        self.layer4 = ResidualBlock(512, 512, stride=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)

        B, C, G, _ = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, G * G, C)
        return self.fc(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER LÉGER — inchangé
# ============================================================================

class EncoderCNNLite(nn.Module):
    """Version allégée pour développement rapide."""

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        x = self.features(images)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DES ENCODEURS")
    print("="*70)

    batch = torch.randn(2, 3, 224, 224)

    encoders = [
        ("EncoderCNNLite    (dev)",                  EncoderCNNLite(512)),
        ("EncoderCNN        (résiduel, global)",      EncoderCNN(512)),
        ("EncoderSpatial    (résiduel, 7×7=49)",      EncoderSpatial(512, grid_size=7)),
        ("EncoderDenseNet   (DenseNet-121, 7×7=49)",  EncoderDenseNet(512, grid_size=7)),
    ]

    for name, enc in encoders:
        out = enc(batch)
        print(f"\n{name}")
        print(f"  Sortie  : {out.shape}")
        print(f"  Params  : {enc.get_num_params():,}")

    # Vérifier la compatibilité DenseNet avec différentes configs
    print("\n── Variantes DenseNet ──")
    for cfg, label in [
        ((6, 12, 24, 16), "DenseNet-121"),
        ((6, 12, 32, 32), "DenseNet-169"),
    ]:
        enc = EncoderDenseNet(512, block_config=cfg)
        out = enc(batch)
        print(f"  {label} : sortie={out.shape}  params={enc.get_num_params():,}")

    print("\n" + "="*70)
    print("Tous les encodeurs fonctionnent !")
    print("="*70)