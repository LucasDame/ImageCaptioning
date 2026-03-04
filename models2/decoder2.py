"""
Decoder LSTM pour Image Captioning
====================================

Améliorations par rapport à l'original :

1. DecoderLSTM : beam search réel.
   L'original avait une coquille vide qui appelait greedy.
   Le beam search garde les k meilleures hypothèses à chaque pas,
   ce qui améliore les scores BLEU sans aucun réentraînement.

2. DecoderWithAttention : attention visuelle de Bahdanau (Show-Attend-Tell).
   À chaque pas, le decoder recalcule un vecteur de contexte en pondérant
   les régions spatiales de l'image → le modèle "regarde" les bonnes zones
   au bon moment.
   Requiert EncoderSpatial comme encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# DECODER ORIGINAL AVEC BEAM SEARCH RÉEL
# ============================================================================

class DecoderLSTM(nn.Module):
    """
    Decoder LSTM avec beam search fonctionnel.
    API identique à l'original → remplacement direct sans modifier train.py.
    """

    def __init__(self, feature_dim, embedding_dim, hidden_dim,
                 vocab_size, num_layers=1, dropout=0.5):
        super().__init__()

        self.feature_dim   = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.vocab_size    = vocab_size
        self.num_layers    = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden(self, features):
        h0 = self.feature_projection(features).unsqueeze(0)
        h0 = h0.repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, features, captions):
        """Teacher forcing — inchangé."""
        hidden     = self.init_hidden(features)
        embeddings = self.dropout(self.embedding(captions))
        lstm_out, _ = self.lstm(embeddings, hidden)
        return self.fc(self.dropout(lstm_out))

    def generate(self, features, max_length=20, start_token=1, end_token=2):
        """Greedy search — inchangé."""
        batch_size = features.size(0)
        hidden     = self.init_hidden(features)
        inputs     = torch.full((batch_size, 1), start_token,
                                dtype=torch.long, device=features.device)
        generated  = []

        for _ in range(max_length):
            emb              = self.embedding(inputs)
            lstm_out, hidden = self.lstm(emb, hidden)
            predicted        = self.fc(lstm_out).argmax(dim=2)
            generated.append(predicted)
            inputs = predicted
            if (predicted == end_token).all():
                break

        return torch.cat(generated, dim=1)

    def generate_beam_search(self, features, beam_width=3,
                             max_length=20, start_token=1, end_token=2):
        """
        Beam search réel (l'original appelait simplement greedy).

        À chaque pas, on garde les beam_width hypothèses avec le meilleur
        score cumulé (log-probabilité). Le score final est normalisé par
        la longueur pour éviter de favoriser les séquences courtes.

        Args:
            features   : (1, feature_dim) — une image à la fois
            beam_width : nombre d'hypothèses conservées (3 est un bon défaut)
            max_length : longueur maximale
            start_token: idx de <START>
            end_token  : idx de <END>

        Returns:
            torch.Tensor : (1, seq_len)
        """
        device = features.device
        hidden = self.init_hidden(features)

        # Chaque beam : (score_cumulé, tokens, hidden_state)
        beams     = [(0.0, [start_token], hidden)]
        completed = []

        for _ in range(max_length):
            new_beams = []

            for score, tokens, h in beams:
                if tokens[-1] == end_token:
                    completed.append((score, tokens))
                    continue

                inp = torch.tensor([[tokens[-1]]], dtype=torch.long,
                                   device=device)
                emb             = self.embedding(inp)
                lstm_out, h_new = self.lstm(emb, h)
                log_probs       = F.log_softmax(
                    self.fc(lstm_out.squeeze(1)), dim=-1
                )
                topk_lp, topk_ids = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    new_beams.append((
                        score + topk_lp[0, k].item(),
                        tokens + [topk_ids[0, k].item()],
                        h_new
                    ))

            if not new_beams:
                break

            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        for score, tokens, _ in beams:
            completed.append((score, tokens))

        # Meilleure hypothèse, normalisée par la longueur
        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return torch.tensor([best[1]], dtype=torch.long, device=device)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MÉCANISME D'ATTENTION DE BAHDANAU
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Attention additive (Bahdanau, 2015).

    Pour chaque pas t du decoder :
      energy_i = v · tanh(W_enc·f_i + W_dec·h_t)
      alpha_i  = softmax(energy_i)
      context  = Σ alpha_i · f_i

    où f_i sont les features de la région i de l'image,
    h_t est le hidden state courant du LSTM.
    """

    def __init__(self, feature_dim, hidden_dim, attention_dim=256):
        super().__init__()
        self.W_enc = nn.Linear(feature_dim, attention_dim)
        self.W_dec = nn.Linear(hidden_dim,  attention_dim)
        self.v     = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        """
        Args:
            features : (B, num_pixels, feature_dim)
            hidden   : (B, hidden_dim)

        Returns:
            context  : (B, feature_dim)
            alpha    : (B, num_pixels)  — utile pour visualiser l'attention
        """
        enc_out = self.W_enc(features)                       # (B, P, att_dim)
        dec_out = self.W_dec(hidden).unsqueeze(1)            # (B, 1, att_dim)
        energy  = self.v(torch.tanh(enc_out + dec_out))      # (B, P, 1)
        alpha   = F.softmax(energy.squeeze(2), dim=1)        # (B, P)
        context = (alpha.unsqueeze(2) * features).sum(dim=1) # (B, feature_dim)
        return context, alpha


# ============================================================================
# DECODER AVEC ATTENTION VISUELLE
# ============================================================================

class DecoderWithAttention(nn.Module):
    """
    Decoder LSTM avec attention de Bahdanau.

    À chaque pas de génération :
      1. BahdanauAttention calcule un contexte visuel pondéré
      2. context + embedding → LSTMCell → prédiction

    Requiert EncoderSpatial (features spatiales) comme encoder.
    Beam search est également implémenté.
    """

    def __init__(self, feature_dim, embedding_dim, hidden_dim,
                 vocab_size, attention_dim=256, dropout=0.5):
        super().__init__()

        self.feature_dim   = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.vocab_size    = vocab_size
        self.num_layers    = 1  # LSTMCell = 1 couche

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(feature_dim, hidden_dim, attention_dim)

        # Initialisation h0 et c0 depuis la moyenne des features spatiales
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # LSTMCell : input = embedding + contexte visuel
        self.lstm    = nn.LSTMCell(embedding_dim + feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden(self, features):
        """
        Initialise h0 et c0 depuis la moyenne des features spatiales.
        features : (B, num_pixels, feature_dim)
        """
        mean_feat = features.mean(dim=1)        # (B, feature_dim)
        h = torch.tanh(self.init_h(mean_feat))  # (B, hidden_dim)
        c = torch.tanh(self.init_c(mean_feat))  # (B, hidden_dim)
        return h, c

    def forward(self, features, captions):
        """
        Teacher forcing avec attention.

        Args:
            features : (B, num_pixels, feature_dim)  — de EncoderSpatial
            captions : (B, seq_len)

        Returns:
            outputs  : (B, seq_len, vocab_size)
        """
        B, seq_len = captions.shape
        embeddings = self.dropout(self.embedding(captions))  # (B, T, emb_dim)
        h, c       = self.init_hidden(features)

        outputs = []
        for t in range(seq_len):
            context, _  = self.attention(features, h)           # (B, feat_dim)
            lstm_input  = torch.cat([embeddings[:, t], context], dim=1)
            h, c        = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(self.dropout(h)).unsqueeze(1))

        return torch.cat(outputs, dim=1)                         # (B, T, vocab)

    def generate(self, features, max_length=20, start_token=1, end_token=2):
        """Greedy avec attention."""
        B    = features.size(0)
        h, c = self.init_hidden(features)
        inp  = torch.full((B,), start_token, dtype=torch.long,
                          device=features.device)
        generated = []

        for _ in range(max_length):
            emb         = self.embedding(inp)
            context, _  = self.attention(features, h)
            h, c        = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            predicted   = self.fc(h).argmax(dim=1)
            generated.append(predicted.unsqueeze(1))
            inp = predicted
            if (predicted == end_token).all():
                break

        return torch.cat(generated, dim=1)

    def generate_beam_search(self, features, beam_width=3,
                             max_length=20, start_token=1, end_token=2):
        """Beam search avec attention — une image à la fois."""
        device   = features.device
        h, c     = self.init_hidden(features)
        beams    = [(0.0, [start_token], h[0], c[0])]
        completed = []

        for _ in range(max_length):
            new_beams = []
            for score, tokens, bh, bc in beams:
                if tokens[-1] == end_token:
                    completed.append((score, tokens))
                    continue

                inp        = torch.tensor([tokens[-1]], dtype=torch.long,
                                          device=device)
                emb        = self.embedding(inp)
                ctx, _     = self.attention(features, bh.unsqueeze(0))
                bh_new, bc_new = self.lstm(
                    torch.cat([emb, ctx], dim=1),
                    (bh.unsqueeze(0), bc.unsqueeze(0))
                )
                bh_new = bh_new.squeeze(0)
                bc_new = bc_new.squeeze(0)

                log_probs         = F.log_softmax(self.fc(bh_new), dim=-1)
                topk_lp, topk_ids = log_probs.topk(beam_width)

                for k in range(beam_width):
                    new_beams.append((
                        score + topk_lp[k].item(),
                        tokens + [topk_ids[k].item()],
                        bh_new, bc_new
                    ))

            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        for score, tokens, _, _ in beams:
            completed.append((score, tokens))

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return torch.tensor([best[1]], dtype=torch.long, device=device)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DES DECODERS")
    print("="*70)

    feat_dim, emb_dim, hid_dim, vocab = 512, 256, 512, 5000
    B, T = 2, 12

    print("\n[DecoderLSTM — greedy + beam search]")
    dec   = DecoderLSTM(feat_dim, emb_dim, hid_dim, vocab)
    feats = torch.randn(B, feat_dim)
    caps  = torch.randint(0, vocab, (B, T))
    print(f"  Forward       : {dec(feats, caps).shape}")
    print(f"  Greedy        : {dec.generate(feats[:1], max_length=8).shape}")
    print(f"  Beam (w=3)    : {dec.generate_beam_search(feats[:1], beam_width=3, max_length=8).shape}")

    print("\n[DecoderWithAttention — greedy + beam search]")
    dec_att  = DecoderWithAttention(feat_dim, emb_dim, hid_dim, vocab)
    feats_sp = torch.randn(B, 49, feat_dim)  # 7×7 spatial
    print(f"  Forward       : {dec_att(feats_sp, caps).shape}")
    print(f"  Greedy        : {dec_att.generate(feats_sp[:1], max_length=8).shape}")
    print(f"  Beam (w=3)    : {dec_att.generate_beam_search(feats_sp[:1], beam_width=3, max_length=8).shape}")

    print(f"\nParams DecoderLSTM          : {dec.get_num_params():,}")
    print(f"Params DecoderWithAttention : {dec_att.get_num_params():,}")

    print("\n" + "="*70)
    print("Tous les decoders fonctionnent !")
    print("="*70)
