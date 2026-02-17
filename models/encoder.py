"""
Encoder CNN pour Image Captioning
==================================

Encoder basé sur une architecture CNN from scratch.
Extrait les features visuelles d'une image.
"""

import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    """
    Encoder CNN qui extrait les features d'une image
    
    Architecture:
    - Plusieurs blocs convolutionnels
    - Chaque bloc: Conv2D → BatchNorm → ReLU → MaxPool
    - Feature extraction finale
    """
    
    def __init__(self, feature_dim=512):
        """
        Args:
            feature_dim (int): Dimension du vecteur de features en sortie
                              Ce vecteur sera donné au decoder
        """
        super(EncoderCNN, self).__init__()
        
        self.feature_dim = feature_dim
        
        # ================================================================
        # BLOC 1: 3 → 64 channels
        # ================================================================
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 → 112x112
        )
        
        # ================================================================
        # BLOC 2: 64 → 128 channels
        # ================================================================
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 → 56x56
        )
        
        # ================================================================
        # BLOC 3: 128 → 256 channels
        # ================================================================
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 → 28x28
        )
        
        # ================================================================
        # BLOC 4: 256 → 512 channels
        # ================================================================
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14
        )
        
        # ================================================================
        # BLOC 5: 512 → 512 channels
        # ================================================================
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 → 7x7
        )
        
        # ================================================================
        # ADAPTIVE POOLING: 7x7 → 1x1
        # ================================================================
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ================================================================
        # FULLY CONNECTED LAYERS
        # ================================================================
        # Après pooling: 512 channels × 1 × 1 = 512 features
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Dropout pour la régularisation
        )
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialise les poids du réseau
        Utilise l'initialisation de Kaiming (He initialization) pour ReLU
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        """
        Forward pass de l'encoder
        
        Args:
            images (torch.Tensor): Batch d'images
                                  Shape: (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Feature vectors
                         Shape: (batch_size, feature_dim)
        """
        # Passer à travers les blocs convolutionnels
        x = self.conv1(images)    # (batch_size, 64, 112, 112)
        x = self.conv2(x)          # (batch_size, 128, 56, 56)
        x = self.conv3(x)          # (batch_size, 256, 28, 28)
        x = self.conv4(x)          # (batch_size, 512, 14, 14)
        x = self.conv5(x)          # (batch_size, 512, 7, 7)
        
        # Adaptive pooling pour obtenir une taille fixe
        x = self.adaptive_pool(x)  # (batch_size, 512, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 512)
        
        # Fully connected layer
        features = self.fc(x)      # (batch_size, feature_dim)
        
        return features
    
    def get_num_params(self):
        """
        Retourne le nombre de paramètres du modèle
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# VERSION LÉGÈRE DE L'ENCODER (pour tester plus rapidement)
# ============================================================================

class EncoderCNNLite(nn.Module):
    """
    Version allégée de l'encoder pour un entraînement plus rapide
    Moins de couches, moins de paramètres
    """
    
    def __init__(self, feature_dim=512):
        super(EncoderCNNLite, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Architecture simplifiée
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 224→112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112→56
            
            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56→28
            
            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28→14
            
            # Bloc 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14→7
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        features = self.fc(x)
        return features
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DE L'ENCODER CNN")
    print("="*70)
    
    # Créer l'encoder
    encoder = EncoderCNN(feature_dim=512)
    encoder_lite = EncoderCNNLite(feature_dim=512)
    
    # Créer une image de test
    batch_size = 4
    test_image = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nImage input shape: {test_image.shape}")
    
    # Forward pass
    print("\n[EncoderCNN]")
    features = encoder(test_image)
    print(f"Features output shape: {features.shape}")
    print(f"Number of parameters: {encoder.get_num_params():,}")
    
    print("\n[EncoderCNNLite]")
    features_lite = encoder_lite(test_image)
    print(f"Features output shape: {features_lite.shape}")
    print(f"Number of parameters: {encoder_lite.get_num_params():,}")
    
    print("\n" + "="*70)
    print("Encoder fonctionne correctement !")
    print("="*70)