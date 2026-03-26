"""
encoder.py — Encodeurs CNN pour Image Captioning COCO
======================================================

Améliorations v4 (inspirées de l'architecture CBAM de Woo et al. 2018) :

1. CBAM (Convolutional Block Attention Module) ajouté après chaque stage
   de TOUS les encodeurs (CNN, Spatial/ResNet, DenseNet).
   → ChannelAttention : repondère les canaux via avg-pool + max-pool → MLP
   → SpatialAttention  : repondère spatialement via concat avg/max sur canaux
   Ces deux modules permettent à l'encodeur de filtrer l'information pertinente
   AVANT de la passer au décodeur, ce qui améliore le signal reçu par l'attention
   de Bahdanau.

2. Grille spatiale 14×14 = 196 patches au lieu de 7×7 = 49 pour EncoderSpatial
   et EncoderDenseNet. L'attention Bahdanau dispose de 4× plus de régions, ce
   qui améliore la précision spatiale (objets petits, relations spatiales).
   → config.py mis à jour : grid_size=14 par défaut pour resnet et densenet.

3. Deux blocs résiduels par stage dans EncoderSpatial et EncoderCNN (au lieu
   d'un seul), ce qui double la profondeur effective pour un coût raisonnable.

4. EncoderCNN (modèle 'cnn') bénéficie aussi du CBAM pour améliorer le vecteur
   global avant de l'envoyer au DecoderLSTM.

Architectures disponibles :
  - EncoderCNN      : résiduel from scratch + CBAM, vecteur global (B, feature_dim)
  - EncoderSpatial  : résiduel from scratch + CBAM, grille 14×14  (B, 196, feature_dim)
  - EncoderDenseNet : DenseNet-121 from scratch + CBAM, grille 14×14 (B, 196, feature_dim)
  - EncoderCNNLite  : version légère pour dev rapide (inchangée)

Aucun modèle pré-entraîné n'est utilisé.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CBAM — Convolutional Block Attention Module (Woo et al. 2018)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Repondération des canaux via avg-pool + max-pool → MLP partagé → sigmoid.

    Chaque canal est pondéré selon son importance globale, indépendamment de
    la position spatiale. Compresse W×H en scalaire, traite via MLP, additionne
    les contributions avg et max avant le sigmoid.

    Args:
        channels  : nombre de canaux d'entrée
        reduction : facteur de réduction du MLP (défaut 16, comme dans le papier)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        # MLP partagé entre avg et max
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        avg_w = self.mlp(self.avg_pool(x))          # (B, C)
        max_w = self.mlp(self.max_pool(x))           # (B, C)
        scale = torch.sigmoid(avg_w + max_w)         # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1) # (B, C, H, W)


class SpatialAttention(nn.Module):
    """
    Repondération spatiale via concaténation avg/max sur les canaux → Conv → sigmoid.

    Compresse le long de l'axe des canaux pour produire une carte 2D,
    qui pondère chaque position spatiale indépendamment.

    Args:
        kernel_size : taille du filtre Conv2D (7 par défaut, comme dans le papier)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_map, _ = x.max(dim=1, keepdim=True)        # (B, 1, H, W)
        concat = torch.cat([avg_map, max_map], dim=1)  # (B, 2, H, W)
        scale  = torch.sigmoid(self.conv(concat))      # (B, 1, H, W)
        return x * scale


class CBAM(nn.Module):
    """
    Bloc CBAM = ChannelAttention → SpatialAttention (ordre du papier original).

    Peut être inséré après n'importe quel bloc convolutif sans changer les
    dimensions d'entrée/sortie.

    Args:
        channels    : nombre de canaux du feature map en entrée
        reduction   : facteur de réduction pour ChannelAttention (défaut 16)
        kernel_size : taille du filtre pour SpatialAttention (défaut 7)
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ============================================================================
# BLOC RÉSIDUEL (pour EncoderCNN / EncoderSpatial)
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Bloc Conv→BN→ReLU→Conv→BN avec connexion résiduelle.

    Utilisé par EncoderCNN et EncoderSpatial.
    Le shortcut utilise une Conv1×1 si les dimensions changent (stride ou canaux).
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

    Args:
        in_channels  : canaux d'entrée (64 + i*growth_rate pour la couche i)
        growth_rate  : k dans l'article DenseNet — canaux produits par cette couche
        dropout      : dropout appliqué après la conv 3×3
    """
    def __init__(self, in_channels, growth_rate, dropout=0.0):
        super().__init__()
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
        return torch.cat([x, self.drop(self.layer(x))], dim=1)


class _DenseBlock(nn.Module):
    """
    Bloc de num_layers couches denses empilées.

    out_channels = in_channels + num_layers * growth_rate
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
# ENCODER CNN (vecteur global) — avec CBAM
# ============================================================================

class EncoderCNN(nn.Module):
    """
    CNN from scratch avec blocs résiduels + CBAM après chaque stage.

    Architecture :
      Stem  : Conv7×7 stride 2 → BN → ReLU → MaxPool   224→56
      Stage1: 2× ResidualBlock(64  → 128, stride 2) + CBAM   56→28
      Stage2: 2× ResidualBlock(128 → 256, stride 2) + CBAM   28→14
      Stage3: 2× ResidualBlock(256 → 512, stride 2) + CBAM   14→7
      Stage4: 2× ResidualBlock(512 → 512, stride 1) + CBAM    7→7
      AdaptiveAvgPool(1,1) → flatten → Linear → ReLU → Dropout

    Le CBAM repondère canaux et positions spatiales après chaque stage,
    ce qui améliore la qualité du vecteur global envoyé au DecoderLSTM.

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

        # Deux blocs par stage (au lieu d'un) + CBAM
        self.layer1 = nn.Sequential(ResidualBlock(64,  128, stride=2),
                                    ResidualBlock(128, 128, stride=1))
        self.cbam1  = CBAM(128)

        self.layer2 = nn.Sequential(ResidualBlock(128, 256, stride=2),
                                    ResidualBlock(256, 256, stride=1))
        self.cbam2  = CBAM(256)

        self.layer3 = nn.Sequential(ResidualBlock(256, 512, stride=2),
                                    ResidualBlock(512, 512, stride=1))
        self.cbam3  = CBAM(512)

        self.layer4 = nn.Sequential(ResidualBlock(512, 512, stride=1),
                                    ResidualBlock(512, 512, stride=1))
        self.cbam4  = CBAM(512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER SPATIAL RÉSIDUEL + CBAM (pour l'attention — grille 14×14)
# ============================================================================

class EncoderSpatial(nn.Module):
    """
    CNN résiduel from scratch + CBAM, retourne une grille spatiale 14×14.

    Par rapport à la v3 :
      - 2 blocs résiduels par stage (au lieu de 1) : profondeur doublée
      - CBAM après chaque stage : filtrage canal + spatial avant l'attention
      - Grille 14×14 = 196 patches (au lieu de 7×7 = 49) : 4× plus de résolution
        spatiale, l'attention Bahdanau dispose de plus de régions à cibler

    Architecture :
      Stem  : Conv7×7/2 → BN → ReLU → MaxPool   224→56
      Stage1: 2×ResBlock(64  → 128, s2) + CBAM    56→28
      Stage2: 2×ResBlock(128 → 256, s2) + CBAM    28→14
      Stage3: 2×ResBlock(256 → 512, s1) + CBAM    14→14   ← stride 1 (pas de downsampling)
      Stage4: 2×ResBlock(512 → 512, s1) + CBAM    14→14   ← stride 1
      AdaptiveAvgPool(14,14) → projection Linear pixelwise

    Sortie : (batch_size, 196, feature_dim)
    """

    def __init__(self, feature_dim=512, grid_size=14):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size   = grid_size

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 1 & 2 : avec stride=2 pour descendre de 56→28→14
        self.layer1 = nn.Sequential(ResidualBlock(64,  128, stride=2),
                                    ResidualBlock(128, 128, stride=1))
        self.cbam1  = CBAM(128)

        self.layer2 = nn.Sequential(ResidualBlock(128, 256, stride=2),
                                    ResidualBlock(256, 256, stride=1))
        self.cbam2  = CBAM(256)

        # Stage 3 & 4 : stride=1, on reste à 14×14 pour préserver la résolution
        self.layer3 = nn.Sequential(ResidualBlock(256, 512, stride=1),
                                    ResidualBlock(512, 512, stride=1))
        self.cbam3  = CBAM(512)

        self.layer4 = nn.Sequential(ResidualBlock(512, 512, stride=1),
                                    ResidualBlock(512, 512, stride=1))
        self.cbam4  = CBAM(512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Projection pixelwise : appliquée indépendamment sur chacune des 196 régions
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        x = self.stem(images)                   # (B,  64, 56, 56)
        x = self.cbam1(self.layer1(x))          # (B, 128, 28, 28)
        x = self.cbam2(self.layer2(x))          # (B, 256, 14, 14)
        x = self.cbam3(self.layer3(x))          # (B, 512, 14, 14)
        x = self.cbam4(self.layer4(x))          # (B, 512, 14, 14)
        x = self.adaptive_pool(x)               # (B, 512, G, G)

        B, C, G, _ = x.shape
        x = x.permute(0, 2, 3, 1)              # (B, G, G, C)
        x = x.reshape(B, G * G, C)             # (B, G², C)
        return self.fc(x)                       # (B, G², feature_dim)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER DENSENET SPATIAL + CBAM (grille 14×14)
# ============================================================================

class EncoderDenseNet(nn.Module):
    """
    DenseNet-121 from scratch + CBAM après chaque DenseBlock, grille 14×14.

    Par rapport à la v3 :
      - CBAM ajouté après chaque DenseBlock (avant la TransitionLayer)
      - Grille 14×14 = 196 patches (au lieu de 7×7 = 49)
        → la Transition3 est supprimée pour conserver la résolution 14×14
        → le DenseBlock4 opère directement sur 14×14

    Architecture (DenseNet-121, growth_rate=32) :
      Stem          : Conv7×7/2 → BN → ReLU → MaxPool    224 → 56
      DenseBlock1   :  6 couches → CBAM                   56  → 56   ch→256
      Transition1   : θ=0.5, AvgPool/2                    56  → 28   ch→128
      DenseBlock2   : 12 couches → CBAM                   28  → 28   ch→512
      Transition2   : θ=0.5, AvgPool/2                    28  → 14   ch→256
      DenseBlock3   : 24 couches → CBAM                   14  → 14   ch→1024
      Transition3   : θ=0.5, Conv1×1 SANS AvgPool         14  → 14   ch→512  ← MODIFIÉ
      DenseBlock4   : 16 couches → CBAM                   14  → 14   ch→1024
      BN → ReLU → AdaptiveAvgPool(14,14) → projection Linear pixelwise

    Sortie : (batch_size, 196, feature_dim)

    Args:
        feature_dim  : dimension de projection finale
        grid_size    : taille de la grille spatiale (14 par défaut → 196 régions)
        growth_rate  : k — 32 pour DenseNet-121
        compression  : θ dans les transitions — 0.5 par défaut
        dropout      : dropout dans les DenseLayers (0.0 = désactivé)
        block_config : (6, 12, 24, 16) = DenseNet-121
    """

    def __init__(self, feature_dim=512, grid_size=14,
                 growth_rate=32, compression=0.5, dropout=0.0,
                 block_config=(6, 12, 24, 16)):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size   = grid_size

        # ── Stem ──────────────────────────────────────────────────────────────
        num_init_features = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7,
                      stride=2, padding=3, bias=False),          # 224 → 112
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # 112 → 56
        )

        # ── DenseBlocks + Transitions + CBAM ──────────────────────────────────
        # La logique change pour le dernier bloc : on supprime l'AvgPool de la
        # dernière Transition afin de rester à 14×14 (grille cible).
        ch = num_init_features
        self.dense_blocks   = nn.ModuleList()
        self.cbam_blocks    = nn.ModuleList()
        self.transitions    = nn.ModuleList()
        n_blocks = len(block_config)

        for i, num_layers in enumerate(block_config):
            # DenseBlock i
            block = _DenseBlock(ch, num_layers, growth_rate, dropout)
            self.dense_blocks.append(block)
            ch = block.out_channels

            # CBAM après chaque DenseBlock
            self.cbam_blocks.append(CBAM(ch))

            # Transition après tous les blocs sauf le dernier
            if i < n_blocks - 1:
                is_last_transition = (i == n_blocks - 2)  # avant le dernier bloc

                if is_last_transition:
                    # Transition sans AvgPool pour conserver 14×14
                    out_ch = int(ch * compression)
                    trans = nn.Sequential(
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch, out_ch, kernel_size=1, bias=False),
                        # Pas d'AvgPool : on reste à 14×14
                    )
                    trans.out_channels = out_ch  # attribut pour compatibilité
                    self.transitions.append(trans)
                    ch = out_ch
                else:
                    trans = _TransitionLayer(ch, compression)
                    self.transitions.append(trans)
                    ch = trans.out_channels

        self.final_ch = ch   # 1024 pour DenseNet-121

        # ── BN final + pool spatial ────────────────────────────────────────────
        self.final_norm = nn.Sequential(
            nn.BatchNorm2d(self.final_ch),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # ── Projection pixelwise ───────────────────────────────────────────────
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
        x = self.stem(images)                        # (B, 64, 56, 56)

        n_transitions = len(self.transitions)
        for i, (block, cbam) in enumerate(zip(self.dense_blocks, self.cbam_blocks)):
            x = block(x)
            x = cbam(x)
            if i < n_transitions:
                x = self.transitions[i](x)

        x = self.final_norm(x)                       # (B, 1024, 14, 14)
        x = self.adaptive_pool(x)                    # (B, 1024, G, G)

        B, C, G, _ = x.shape
        x = x.permute(0, 2, 3, 1)                   # (B, G, G, C)
        x = x.reshape(B, G * G, C)                  # (B, G², C)
        return self.fc(x)                            # (B, G², feature_dim)

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
    print("TEST DES ENCODEURS v4 (CBAM + grille 14×14)")
    print("="*70)

    batch = torch.randn(2, 3, 224, 224)

    encoders = [
        ("EncoderCNNLite    (dev, inchangé)",            EncoderCNNLite(512)),
        ("EncoderCNN        (résiduel + CBAM, global)",  EncoderCNN(512)),
        ("EncoderSpatial    (résiduel + CBAM, 14×14)",   EncoderSpatial(512, grid_size=14)),
        ("EncoderDenseNet   (DenseNet-121 + CBAM, 14×14)", EncoderDenseNet(512, grid_size=14)),
    ]

    for name, enc in encoders:
        out = enc(batch)
        print(f"\n{name}")
        print(f"  Sortie  : {out.shape}")
        print(f"  Params  : {enc.get_num_params():,}")

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