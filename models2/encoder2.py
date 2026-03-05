"""
Encoder CNN pour Image Captioning
==================================

Architectures disponibles :
  - EncoderCNNLite  : version légère pour le développement rapide (inchangée)
  - EncoderCNN      : CNN from scratch avec blocs résiduels
  - EncoderSpatial  : retourne une carte 7×7 pour l'attention visuelle
                      (à utiliser avec DecoderWithAttention)

Aucun modèle pré-entraîné n'est utilisé.
"""

import torch
import torch.nn as nn


# ============================================================================
# BLOC RÉSIDUEL
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Bloc Conv→BN→ReLU→Conv→BN avec connexion résiduelle.

    Si les dimensions d'entrée et de sortie diffèrent (changement de channels
    ou stride > 1), une projection 1×1 est ajoutée sur le chemin résiduel.

    Pourquoi ?
    L'EncoderCNN original empilait des blocs sans résidu : les gradients
    s'atténuaient dans les premières couches (vanishing gradient).
    Avec le shortcut, le gradient remonte directement jusqu'au début du réseau.
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

        # Projection si les dimensions changent
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
# ENCODER RÉSIDUEL (from scratch)
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 224→56
        )

        self.layer1 = ResidualBlock(64,  128, stride=2)   # 56→28
        self.layer2 = ResidualBlock(128, 256, stride=2)   # 28→14
        self.layer3 = ResidualBlock(256, 512, stride=2)   # 14→7
        self.layer4 = ResidualBlock(512, 512, stride=1)   #  7→7

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, images):
        x = self.stem(images)      # (B,  64, 56, 56)
        x = self.layer1(x)         # (B, 128, 28, 28)
        x = self.layer2(x)         # (B, 256, 14, 14)
        x = self.layer3(x)         # (B, 512,  7,  7)
        x = self.layer4(x)         # (B, 512,  7,  7)
        x = self.adaptive_pool(x)  # (B, 512,  1,  1)
        x = x.view(x.size(0), -1)  # (B, 512)
        return self.fc(x)          # (B, feature_dim)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER SPATIAL (pour l'attention — from scratch)
# ============================================================================

class EncoderSpatial(nn.Module):
    """
    Mêmesize × grid_siz architecture résiduelle que EncoderCNN, mais sans le pooling global.
    Retourne une carte spatiale (grid_e) de features.

    Utilisé avec DecoderWithAttention : le decoder peut ainsi "regarder"
    différentes régions de l'image à chaque pas de génération.

    Sortie : (batch_size, grid_size², feature_dim)
             ex: (B, 49, 512) pour grid_size=7
    """

    def __init__(self, feature_dim=512, grid_size=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size   = grid_size

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 224→56
        )

        self.layer1 = ResidualBlock(64,  128, stride=2)   # 56→28
        self.layer2 = ResidualBlock(128, 256, stride=2)   # 28→14
        self.layer3 = ResidualBlock(256, 512, stride=2)   # 14→7
        self.layer4 = ResidualBlock(512, 512, stride=1)   #  7→7

        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Projection pixel-wise appliquée indépendamment sur chaque région
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        x = self.stem(images)          # (B,  64, 56, 56)
        x = self.layer1(x)             # (B, 128, 28, 28)
        x = self.layer2(x)             # (B, 256, 14, 14)
        x = self.layer3(x)             # (B, 512,  7,  7)
        x = self.layer4(x)             # (B, 512,  7,  7)
        x = self.adaptive_pool(x)      # (B, 512,  G,  G)

        B, C, G, _ = x.shape
        x = x.permute(0, 2, 3, 1)     # (B, G, G, C)
        x = x.reshape(B, G * G, C)    # (B, G², C)
        return self.fc(x)              # (B, G², feature_dim)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ENCODER LÉGER (inchangé)
# ============================================================================

class EncoderCNNLite(nn.Module):
    """
    Version allégée pour développement rapide. Inchangée.
    """

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

    for name, enc in [
        ("EncoderCNNLite  (dev)",               EncoderCNNLite(512)),
        ("EncoderCNN      (résiduel)",           EncoderCNN(512)),
        ("EncoderSpatial  (attention, 7×7=49)",  EncoderSpatial(512, grid_size=7)),
    ]:
        out = enc(batch)
        print(f"\n{name}")
        print(f"  Sortie : {out.shape}  |  Params : {enc.get_num_params():,}")

    print("\n" + "="*70)
    print("Tous les encodeurs fonctionnent !")
    print("="*70)
