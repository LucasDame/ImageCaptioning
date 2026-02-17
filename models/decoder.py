"""
Decoder LSTM pour Image Captioning
===================================

Decoder qui génère une caption mot par mot à partir des features d'image.
Utilise des embeddings et un LSTM.
"""

import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    """
    Decoder LSTM qui génère des captions
    
    Architecture:
    Image features → Projection → Combine with word embeddings → LSTM → Output
    """
    
    def __init__(self, feature_dim, embedding_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.5):
        """
        Args:
            feature_dim (int): Dimension des features de l'encoder
            embedding_dim (int): Dimension des word embeddings
            hidden_dim (int): Dimension du hidden state du LSTM
            vocab_size (int): Taille du vocabulaire
            num_layers (int): Nombre de couches LSTM
            dropout (float): Taux de dropout
        """
        super(DecoderLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # ================================================================
        # EMBEDDING LAYER
        # ================================================================
        # Convertit les indices de mots en vecteurs denses
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Index du token <PAD>
        )
        
        # ================================================================
        # PROJECTION DES FEATURES
        # ================================================================
        # Projette les features de l'image dans l'espace du LSTM
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # ================================================================
        # LSTM
        # ================================================================
        # Input: embeddings (embedding_dim)
        # Hidden state: hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ================================================================
        # DROPOUT
        # ================================================================
        self.dropout = nn.Dropout(dropout)
        
        # ================================================================
        # OUTPUT LAYER
        # ================================================================
        # Projette le hidden state du LSTM vers le vocabulaire
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialise les poids du decoder
        """
        # Initialiser les embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Garder le padding embedding à zéro
        self.embedding.weight.data[0].fill_(0)
        
        # Initialiser les couches linéaires
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def init_hidden(self, features):
        """
        Initialise le hidden state du LSTM avec les features de l'image
        
        Args:
            features (torch.Tensor): Features de l'image
                                    Shape: (batch_size, feature_dim)
        
        Returns:
            tuple: (h0, c0) - hidden state et cell state initiaux
                   Shape: (num_layers, batch_size, hidden_dim)
        """
        batch_size = features.size(0)
        
        # Projeter les features dans l'espace du hidden state
        h0 = self.feature_projection(features)  # (batch_size, hidden_dim)
        h0 = h0.unsqueeze(0)  # (1, batch_size, hidden_dim)
        
        # Répéter pour chaque couche du LSTM
        h0 = h0.repeat(self.num_layers, 1, 1)  # (num_layers, batch_size, hidden_dim)
        
        # Initialiser le cell state à zéro
        c0 = torch.zeros_like(h0)
        
        return h0, c0
    
    def forward(self, features, captions):
        """
        Forward pass du decoder (mode entraînement avec teacher forcing)
        
        Args:
            features (torch.Tensor): Features de l'encoder
                                    Shape: (batch_size, feature_dim)
            captions (torch.Tensor): Captions (indices de mots)
                                    Shape: (batch_size, seq_len)
                                    Contient [<START>, word1, word2, ..., <END>, <PAD>, ...]
        
        Returns:
            torch.Tensor: Predictions pour chaque mot
                         Shape: (batch_size, seq_len, vocab_size)
        """
        batch_size = features.size(0)
        
        # Initialiser le hidden state avec les features de l'image
        hidden = self.init_hidden(features)  # (h0, c0)
        
        # Convertir les indices de mots en embeddings
        embeddings = self.embedding(captions)  # (batch_size, seq_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        
        # Passer à travers le LSTM
        lstm_out, _ = self.lstm(embeddings, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Prédire les mots
        outputs = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
    def generate(self, features, max_length=20, start_token=1, end_token=2):
        """
        Génère une caption pour une image (mode inférence)
        Utilise greedy search (prend toujours le mot le plus probable)
        
        Args:
            features (torch.Tensor): Features de l'image
                                    Shape: (1, feature_dim) ou (batch_size, feature_dim)
            max_length (int): Longueur maximale de la caption
            start_token (int): Index du token <START>
            end_token (int): Index du token <END>
        
        Returns:
            torch.Tensor: Caption générée (indices)
                         Shape: (batch_size, seq_len)
        """
        batch_size = features.size(0)
        
        # Initialiser le hidden state
        hidden = self.init_hidden(features)
        
        # Commencer avec le token <START>
        inputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=features.device)
        
        # Liste pour stocker les mots générés
        generated_captions = []
        
        # Générer mot par mot
        for _ in range(max_length):
            # Convertir en embedding
            embeddings = self.embedding(inputs)  # (batch_size, 1, embedding_dim)
            
            # LSTM step
            lstm_out, hidden = self.lstm(embeddings, hidden)  # lstm_out: (batch_size, 1, hidden_dim)
            
            # Prédire le prochain mot
            outputs = self.fc(lstm_out)  # (batch_size, 1, vocab_size)
            
            # Prendre le mot le plus probable (greedy search)
            predicted = outputs.argmax(dim=2)  # (batch_size, 1)
            
            # Ajouter à la liste
            generated_captions.append(predicted)
            
            # Utiliser le mot prédit comme prochain input
            inputs = predicted
            
            # Arrêter si on génère <END> pour tous les samples du batch
            if (predicted == end_token).all():
                break
        
        # Concaténer tous les mots générés
        generated_captions = torch.cat(generated_captions, dim=1)  # (batch_size, seq_len)
        
        return generated_captions
    
    def generate_beam_search(self, features, beam_width=3, max_length=20, start_token=1, end_token=2):
        """
        Génère une caption en utilisant beam search (meilleure qualité que greedy)
        
        Args:
            features (torch.Tensor): Features de l'image, shape (1, feature_dim)
            beam_width (int): Largeur du beam
            max_length (int): Longueur maximale
            start_token (int): Index de <START>
            end_token (int): Index de <END>
        
        Returns:
            torch.Tensor: Meilleure caption trouvée
        """
        # Note: Beam search est plus complexe, cette version est simplifiée
        # Pour l'instant, on utilise juste greedy search
        # Vous pouvez implémenter une vraie beam search comme amélioration
        return self.generate(features, max_length, start_token, end_token)
    
    def get_num_params(self):
        """
        Retourne le nombre de paramètres du modèle
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DU DECODER LSTM")
    print("="*70)
    
    # Paramètres
    feature_dim = 512
    embedding_dim = 256
    hidden_dim = 512
    vocab_size = 5000
    batch_size = 4
    seq_len = 15
    
    # Créer le decoder
    decoder = DecoderLSTM(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_layers=1,
        dropout=0.5
    )
    
    print(f"\nDécoder créé avec {decoder.get_num_params():,} paramètres")
    
    # Test 1: Forward pass (entraînement)
    print("\n[TEST 1] Forward pass (entraînement)")
    features = torch.randn(batch_size, feature_dim)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"  Features shape: {features.shape}")
    print(f"  Captions shape: {captions.shape}")
    
    outputs = decoder(features, captions)
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Test 2: Generation (inférence)
    print("\n[TEST 2] Caption generation (inférence)")
    test_features = torch.randn(1, feature_dim)
    generated = decoder.generate(test_features, max_length=20)
    
    print(f"  Generated caption shape: {generated.shape}")
    print(f"  Generated indices: {generated[0].tolist()}")
    
    print("\n" + "="*70)
    print("Decoder fonctionne correctement !")
    print("="*70)