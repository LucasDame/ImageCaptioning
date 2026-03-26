"""
decoder.py — Décodeurs LSTM pour Image Captioning COCO
=======================================================

Améliorations v4 (inspirées de l'architecture du notebook de référence) :

1. DecoderWithAttention — trois nouveautés majeures :
   a) Self-Attention multi-tête causale sur les embeddings (avant la boucle LSTM).
      Un masque causal empêche chaque position de voir les tokens futurs.
      Connexion résiduelle + LayerNorm (style Transformer).
      → Capture les dépendances long-terme entre mots déjà générés.
   b) hidden_dim par défaut porté à 1024 (au lieu de 512).
      La capacité du LSTM est le goulot d'étranglement le plus fréquent.
   c) step() — méthode d'inférence token-par-token qui maintient un cache
      embed_history pour la self-attention (cohérence train / inférence).
      Utilisée par generate(), generate_beam_search(), et les variantes.

2. DecoderLSTM — inchangé (modèle 'cnn', pas d'attention).

3. BahdanauAttention — inchangée (déjà correcte en v3).

Note sur la cohérence train / inférence de la self-attention :
  En entraînement (forward) : toute la séquence en une passe, masque causal.
  En inférence   (step)     : un token à la fois, embed_history accumulé.
  Les deux donnent exactement le même résultat qu'un Transformer causal
  séquentiel, car le masque causal est équivalent à l'accumulation de contexte.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# DECODER ORIGINAL SANS ATTENTION (modèle 'cnn') — inchangé
# ============================================================================

class DecoderLSTM(nn.Module):
    """
    Decoder LSTM avec beam search.
    Utilisé uniquement avec EncoderCNN (vecteur global, pas de grille spatiale).
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
        hidden     = self.init_hidden(features)
        embeddings = self.dropout(self.embedding(captions))
        lstm_out, _ = self.lstm(embeddings, hidden)
        return self.fc(self.dropout(lstm_out))

    def generate(self, features, max_length=20, start_token=1, end_token=2):
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

    def generate_beam_search(self, features, beam_width=5,
                             max_length=20, start_token=1, end_token=2):
        device = features.device
        hidden = self.init_hidden(features)

        beams     = [(0.0, [start_token], hidden)]
        completed = []

        for _ in range(max_length):
            new_beams = []

            for score, tokens, h in beams:
                if tokens[-1] == end_token:
                    completed.append((score, tokens))
                    continue

                inp = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
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

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return torch.tensor([best[1]], dtype=torch.long, device=device)

    def generate_diverse_beam_search(self, features, num_captions=5,
                                     beam_width=5, max_length=20,
                                     start_token=1, end_token=2,
                                     diversity_penalty=0.8):
        device = features.device
        hidden_init = self.init_hidden(features)
        group_beams = [
            [(0.0, [start_token], hidden_init)]
            for _ in range(num_captions)
        ]
        group_completed = [[] for _ in range(num_captions)]

        for _ in range(max_length):
            chosen_tokens_so_far = []

            for g in range(num_captions):
                new_beams_g = []

                for score, tokens, h in group_beams[g]:
                    if tokens[-1] == end_token:
                        group_completed[g].append((score, tokens))
                        continue

                    inp = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                    emb             = self.embedding(inp)
                    lstm_out, h_new = self.lstm(emb, h)
                    log_probs       = F.log_softmax(
                        self.fc(lstm_out.squeeze(1)), dim=-1
                    )

                    log_probs_div = log_probs.clone()
                    for prev_token in chosen_tokens_so_far:
                        log_probs_div[0, prev_token] -= diversity_penalty

                    topk_lp, topk_ids = log_probs_div.topk(beam_width, dim=-1)

                    for k in range(beam_width):
                        tok     = topk_ids[0, k].item()
                        real_lp = log_probs[0, tok].item()
                        new_beams_g.append((
                            score + real_lp,
                            tokens + [tok],
                            h_new
                        ))

                if not new_beams_g:
                    continue

                new_beams_g.sort(key=lambda x: x[0], reverse=True)
                group_beams[g] = new_beams_g[:beam_width]

                if group_beams[g]:
                    head_token = group_beams[g][0][1][-1]
                    chosen_tokens_so_far.append(head_token)

        results = []
        for g in range(num_captions):
            for score, tokens, _ in group_beams[g]:
                group_completed[g].append((score, tokens))
            if group_completed[g]:
                best = max(group_completed[g],
                           key=lambda x: x[0] / max(len(x[1]), 1))
                results.append(best)

        results.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        return [
            torch.tensor([tokens], dtype=torch.long, device=device)
            for _, tokens in results
        ]

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MÉCANISME D'ATTENTION DE BAHDANAU — inchangé
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Attention additive (Bahdanau, 2015).

    energy_i = v · tanh(W_enc·f_i + W_dec·h_t)
    alpha_i  = softmax(energy_i)
    context  = Σ alpha_i · f_i
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
            alpha    : (B, num_pixels)
        """
        enc_out = self.W_enc(features)                       # (B, P, att_dim)
        dec_out = self.W_dec(hidden).unsqueeze(1)            # (B, 1, att_dim)
        energy  = self.v(torch.tanh(enc_out + dec_out))      # (B, P, 1)
        alpha   = F.softmax(energy.squeeze(2), dim=1)        # (B, P)
        context = (alpha.unsqueeze(2) * features).sum(dim=1) # (B, feature_dim)
        return context, alpha


# ============================================================================
# DECODER AVEC ATTENTION + SELF-ATTENTION (modèles 'resnet' et 'densenet')
# ============================================================================

class DecoderWithAttention(nn.Module):
    """
    Décodeur LSTM avec attention de Bahdanau + self-attention multi-tête causale.

    Nouveautés v4 :
      1. Self-attention causale (nn.MultiheadAttention) sur les embeddings
         avant la boucle LSTM. Connexion résiduelle + LayerNorm.
         → Les dépendances long-terme entre mots sont capturées avant le LSTM.
      2. step() : inférence token-par-token avec cache embed_history pour
         reproduire exactement le comportement de la self-attention du forward.
      3. hidden_dim par défaut à 1024 (plus de capacité pour le LSTM).

    Paramètres :
        feature_dim   : dimension des features de l'encodeur (= feature_dim config)
        embedding_dim : dimension des embeddings de mots
        hidden_dim    : dimension cachée du LSTMCell (1024 recommandé)
        vocab_size    : taille du vocabulaire
        attention_dim : dimension de l'espace d'attention Bahdanau
        dropout       : taux de dropout
        num_heads     : têtes de la self-attention (8 recommandé)
        sa_dropout    : dropout dans la self-attention
    """

    def __init__(self, feature_dim, embedding_dim, hidden_dim,
                 vocab_size, attention_dim=256, dropout=0.5,
                 num_heads=8, sa_dropout=0.1):
        super().__init__()

        self.feature_dim   = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.vocab_size    = vocab_size
        self.attention_dim = attention_dim
        self.num_layers    = 1  # LSTMCell = toujours 1 couche

        # ── Embedding ─────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # ── Attention encoder→decoder (Bahdanau) ──────────────────────────────
        self.attention = BahdanauAttention(feature_dim, hidden_dim, attention_dim)

        # ── Self-Attention causale sur les embeddings ──────────────────────────
        # batch_first=True : entrée/sortie en (B, T, E)
        # Le masque causal empêche la position t de voir t+1, t+2, ...
        # Connexion résiduelle + LayerNorm pour la stabilité de l'entraînement.
        self.self_attn     = nn.MultiheadAttention(
            embedding_dim, num_heads,
            dropout=sa_dropout,
            batch_first=True,
        )
        self.sa_layer_norm = nn.LayerNorm(embedding_dim)
        self.sa_dropout    = nn.Dropout(sa_dropout)

        # ── Initialisation état caché depuis la moyenne des features ───────────
        # h ET c sont initialisés avec tanh (au lieu de c=0 en v3).
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # ── LSTMCell : input = embedding (enrichi SA) + contexte visuel ────────
        self.lstm    = nn.LSTMCell(embedding_dim + feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    @staticmethod
    def _causal_mask(T, device):
        """
        Masque carré T×T : True = position interdite (token futur).
        Passé à nn.MultiheadAttention comme attn_mask.
        """
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

    def init_hidden(self, features):
        """
        Initialise h0 et c0 depuis la moyenne des features spatiales.

        Args:
            features : (B, num_pixels, feature_dim)
        Returns:
            h : (B, hidden_dim)
            c : (B, hidden_dim)
        """
        mean_feat = features.mean(dim=1)         # (B, feature_dim)
        h = torch.tanh(self.init_h(mean_feat))   # (B, hidden_dim)
        c = torch.tanh(self.init_c(mean_feat))   # (B, hidden_dim)
        return h, c

    # ── FORWARD (entraînement — teacher forcing, séquence complète) ───────────

    def forward(self, features, captions):
        """
        Teacher forcing avec self-attention causale + attention Bahdanau.

        Args:
            features : (B, num_pixels, feature_dim)
            captions : (B, seq_len)
        Returns:
            outputs  : (B, seq_len, vocab_size)
        """
        B, seq_len = captions.shape
        embeddings = self.dropout(self.embedding(captions))  # (B, T, emb_dim)

        # Self-attention causale sur TOUTE la séquence en une passe
        causal_mask   = self._causal_mask(seq_len, captions.device)
        sa_out, _     = self.self_attn(embeddings, embeddings, embeddings,
                                        attn_mask=causal_mask)
        embeddings    = self.sa_layer_norm(embeddings + self.sa_dropout(sa_out))

        h, c = self.init_hidden(features)
        outputs = []

        for t in range(seq_len):
            context, _  = self.attention(features, h)
            lstm_input  = torch.cat([embeddings[:, t], context], dim=1)
            h, c        = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(self.dropout(h)).unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, T, vocab)

    def forward_with_alphas(self, features, captions):
        """
        Teacher forcing — retourne aussi les poids d'attention pour la
        régularisation doubly stochastic (Xu et al. 2015).

        Returns:
            outputs : (B, seq_len, vocab_size)
            alphas  : (B, seq_len, num_pixels)
        """
        B, seq_len = captions.shape
        embeddings = self.dropout(self.embedding(captions))  # (B, T, emb_dim)

        causal_mask   = self._causal_mask(seq_len, captions.device)
        sa_out, _     = self.self_attn(embeddings, embeddings, embeddings,
                                        attn_mask=causal_mask)
        embeddings    = self.sa_layer_norm(embeddings + self.sa_dropout(sa_out))

        h, c        = self.init_hidden(features)
        outputs     = []
        alphas_list = []

        for t in range(seq_len):
            context, alpha = self.attention(features, h)
            lstm_input     = torch.cat([embeddings[:, t], context], dim=1)
            h, c           = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(self.dropout(h)).unsqueeze(1))
            alphas_list.append(alpha.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)   # (B, T, vocab)
        alphas  = torch.cat(alphas_list, dim=1)  # (B, T, P)
        return outputs, alphas

    # ── STEP (inférence — un token à la fois) ─────────────────────────────────

    def step(self, token_id, hidden, features, embed_history=None):
        """
        Un pas d'inférence auto-regressif avec self-attention sur l'historique.

        Args:
            token_id      : (B, 1) token courant
            hidden        : (h, c) — état LSTMCell courant
            features      : (B, num_pixels, feature_dim) — features de l'encodeur
            embed_history : (B, t, embedding_dim) ou None — embeddings déjà vus
                            Doit être maintenu par l'appelant et concaténé à chaque step.

        Returns:
            logits      : (B, vocab_size)
            hidden      : (h, c) mis à jour
            alpha       : (B, num_pixels) poids d'attention Bahdanau
            embed_cur   : (B, 1, embedding_dim) embedding enrichi par SA
                          → à concaténer à embed_history pour le pas suivant
        """
        h, c = hidden
        emb  = self.embedding(token_id.squeeze(1))  # (B, embedding_dim)
        emb  = emb.unsqueeze(1)                      # (B, 1, embedding_dim)

        # Self-attention sur l'embedding courant + historique
        if embed_history is not None:
            full_seq = torch.cat([embed_history, emb], dim=1)  # (B, t+1, E)
        else:
            full_seq = emb                                       # (B, 1, E)

        T = full_seq.size(1)
        causal_mask = self._causal_mask(T, token_id.device)
        sa_out, _   = self.self_attn(full_seq, full_seq, full_seq,
                                      attn_mask=causal_mask)
        full_seq    = self.sa_layer_norm(full_seq + self.sa_dropout(sa_out))

        # Seul l'embedding du token courant (dernière position) entre dans le LSTM
        embed_cur   = full_seq[:, -1:, :]             # (B, 1, E)
        embed_step  = embed_cur.squeeze(1)             # (B, E)

        # Attention Bahdanau + LSTMCell
        context, alpha = self.attention(features, h)
        lstm_input     = torch.cat([embed_step, context], dim=1)
        h_new, c_new   = self.lstm(lstm_input, (h, c))

        logits = self.fc(self.dropout(h_new))          # (B, vocab_size)
        return logits, (h_new, c_new), alpha, embed_cur

    # ── GÉNÉRATION ─────────────────────────────────────────────────────────────

    def generate(self, features, max_length=20, start_token=1, end_token=2):
        """Greedy avec self-attention + attention Bahdanau."""
        B      = features.size(0)
        device = features.device
        h, c   = self.init_hidden(features)

        inp           = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        embed_history = None
        generated     = []

        for _ in range(max_length):
            logits, (h, c), _, embed_cur = self.step(inp, (h, c), features,
                                                      embed_history)
            embed_history = embed_cur if embed_history is None \
                            else torch.cat([embed_history, embed_cur], dim=1)
            predicted = logits.argmax(dim=1)
            generated.append(predicted.unsqueeze(1))
            inp = predicted.unsqueeze(1)
            if (predicted == end_token).all():
                break

        return torch.cat(generated, dim=1)  # (B, T)

    def generate_beam_search(self, features, beam_width=5,
                             max_length=20, start_token=1, end_token=2):
        """Beam search avec self-attention + attention Bahdanau — une image."""
        device  = features.device
        h, c    = self.init_hidden(features)

        # Chaque beam stocke : (score, tokens, h, c, embed_history)
        beams     = [(0.0, [start_token], h[0], c[0], None)]
        completed = []

        for _ in range(max_length):
            new_beams = []
            for score, tokens, bh, bc, emb_hist in beams:
                if tokens[-1] == end_token:
                    completed.append((score, tokens))
                    continue

                inp   = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                logits, (bh_new, bc_new), _, embed_cur = self.step(
                    inp, (bh.unsqueeze(0), bc.unsqueeze(0)), features, emb_hist
                )
                bh_new = bh_new.squeeze(0)
                bc_new = bc_new.squeeze(0)

                new_hist = embed_cur if emb_hist is None \
                           else torch.cat([emb_hist, embed_cur], dim=1)

                log_probs         = F.log_softmax(logits, dim=-1)
                topk_lp, topk_ids = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    new_beams.append((
                        score + topk_lp[0, k].item(),
                        tokens + [topk_ids[0, k].item()],
                        bh_new, bc_new, new_hist
                    ))

            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        for score, tokens, _, _, _ in beams:
            completed.append((score, tokens))

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return torch.tensor([best[1]], dtype=torch.long, device=device)

    def generate_diverse_beam_search(self, features, num_captions=5,
                                     beam_width=5, max_length=20,
                                     start_token=1, end_token=2,
                                     diversity_penalty=0.8):
        """
        Diverse Beam Search avec self-attention + Bahdanau.
        Génère num_captions captions différentes et pertinentes.
        """
        device         = features.device
        h_init, c_init = self.init_hidden(features)

        # Un faisceau par groupe
        group_beams = [
            [(0.0, [start_token], h_init[0].clone(), c_init[0].clone(), None)]
            for _ in range(num_captions)
        ]
        group_completed = [[] for _ in range(num_captions)]

        for _ in range(max_length):
            chosen_tokens_so_far = []

            for g in range(num_captions):
                new_beams_g = []

                for score, tokens, bh, bc, emb_hist in group_beams[g]:
                    if tokens[-1] == end_token:
                        group_completed[g].append((score, tokens))
                        continue

                    inp = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                    logits, (bh_new, bc_new), _, embed_cur = self.step(
                        inp, (bh.unsqueeze(0), bc.unsqueeze(0)), features, emb_hist
                    )
                    bh_new = bh_new.squeeze(0)
                    bc_new = bc_new.squeeze(0)

                    new_hist = embed_cur if emb_hist is None \
                               else torch.cat([emb_hist, embed_cur], dim=1)

                    log_probs     = F.log_softmax(logits, dim=-1).squeeze(0)
                    log_probs_div = log_probs.clone()
                    for prev_token in chosen_tokens_so_far:
                        log_probs_div[prev_token] -= diversity_penalty

                    topk_lp, topk_ids = log_probs_div.topk(beam_width)

                    for k in range(beam_width):
                        tok     = topk_ids[k].item()
                        real_lp = log_probs[tok].item()
                        new_beams_g.append((
                            score + real_lp,
                            tokens + [tok],
                            bh_new, bc_new, new_hist
                        ))

                if not new_beams_g:
                    continue

                new_beams_g.sort(key=lambda x: x[0], reverse=True)
                group_beams[g] = new_beams_g[:beam_width]

                if group_beams[g]:
                    chosen_tokens_so_far.append(group_beams[g][0][1][-1])

        results = []
        for g in range(num_captions):
            for score, tokens, _, _, _ in group_beams[g]:
                group_completed[g].append((score, tokens))
            if group_completed[g]:
                best = max(group_completed[g],
                           key=lambda x: x[0] / max(len(x[1]), 1))
                results.append(best)

        results.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        return [
            torch.tensor([tokens], dtype=torch.long, device=device)
            for _, tokens in results
        ]

    def generate_with_attention(self, features, max_length=20,
                                start_token=1, end_token=2):
        """Greedy + retourne les poids d'attention Bahdanau à chaque pas."""
        device  = features.device
        h, c    = self.init_hidden(features)
        inp     = torch.tensor([[start_token]], dtype=torch.long, device=device)

        tokens        = []
        alphas        = []
        embed_history = None

        with torch.no_grad():
            for _ in range(max_length):
                logits, (h, c), alpha, embed_cur = self.step(
                    inp, (h, c), features, embed_history
                )
                embed_history = embed_cur if embed_history is None \
                                else torch.cat([embed_history, embed_cur], dim=1)

                predicted = logits.argmax(dim=1)
                tokens.append(predicted.item())
                alphas.append(alpha.squeeze(0))

                if predicted.item() == end_token:
                    break
                inp = predicted.unsqueeze(1)

        alphas = torch.stack(alphas, dim=0)
        return tokens, alphas

    def generate_beam_search_with_attention(self, features, beam_width=5,
                                            max_length=20,
                                            start_token=1, end_token=2):
        """Beam search + retourne les poids d'attention de la meilleure hypothèse."""
        device  = features.device
        h, c    = self.init_hidden(features)

        beams     = [(0.0, [start_token], h[0], c[0], None, [])]
        completed = []

        with torch.no_grad():
            for _ in range(max_length):
                new_beams = []

                for score, tokens, bh, bc, emb_hist, beam_alphas in beams:
                    if tokens[-1] == end_token:
                        completed.append((score, tokens, beam_alphas))
                        continue

                    inp   = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                    logits, (bh_new, bc_new), alpha, embed_cur = self.step(
                        inp, (bh.unsqueeze(0), bc.unsqueeze(0)), features, emb_hist
                    )
                    bh_new  = bh_new.squeeze(0)
                    bc_new  = bc_new.squeeze(0)
                    alpha_t = alpha.squeeze(0)

                    new_hist = embed_cur if emb_hist is None \
                               else torch.cat([emb_hist, embed_cur], dim=1)

                    log_probs         = F.log_softmax(logits, dim=-1)
                    topk_lp, topk_ids = log_probs.topk(beam_width, dim=-1)

                    for k in range(beam_width):
                        new_beams.append((
                            score + topk_lp[0, k].item(),
                            tokens + [topk_ids[0, k].item()],
                            bh_new, bc_new, new_hist,
                            beam_alphas + [alpha_t]
                        ))

                if not new_beams:
                    break
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_width]

            for score, tokens, bh, bc, emb_hist, beam_alphas in beams:
                completed.append((score, tokens, beam_alphas))

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        _, best_tokens, best_alphas = best

        best_tokens = best_tokens[1:]
        alphas = torch.stack(best_alphas, dim=0)
        return best_tokens, alphas

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DES DECODERS v4 (self-attention + hidden_dim=1024)")
    print("="*70)

    feat_dim, emb_dim, hid_dim, vocab = 512, 256, 1024, 5000
    B, T = 2, 12

    print("\n[DecoderLSTM — inchangé]")
    dec   = DecoderLSTM(feat_dim, emb_dim, 512, vocab)
    feats = torch.randn(B, feat_dim)
    caps  = torch.randint(0, vocab, (B, T))
    print(f"  Forward  : {dec(feats, caps).shape}")
    print(f"  Greedy   : {dec.generate(feats[:1], max_length=8).shape}")
    print(f"  Beam(5)  : {dec.generate_beam_search(feats[:1], beam_width=5, max_length=8).shape}")

    print("\n[DecoderWithAttention — self-attention + Bahdanau]")
    dec_att  = DecoderWithAttention(feat_dim, emb_dim, hid_dim, vocab,
                                     num_heads=8, sa_dropout=0.1)
    feats_sp = torch.randn(B, 196, feat_dim)  # grille 14×14

    out = dec_att(feats_sp, caps)
    print(f"  forward()              : {out.shape}")

    out_a, alphas = dec_att.forward_with_alphas(feats_sp, caps)
    print(f"  forward_with_alphas()  : outputs={out_a.shape}, alphas={alphas.shape}")
    assert alphas.shape == (B, T, 196), f"Shape inattendue : {alphas.shape}"

    attention_sum = alphas.sum(dim=1)
    penalty = ((1.0 - attention_sum) ** 2).mean()
    print(f"  Pénalité doubly stoch  : {penalty.item():.6f}")

    print(f"  Greedy                 : {dec_att.generate(feats_sp[:1], max_length=8).shape}")
    print(f"  Beam(5)                : {dec_att.generate_beam_search(feats_sp[:1], beam_width=5, max_length=8).shape}")

    # Test step() cohérence avec forward()
    print("\n  [Cohérence step() vs forward()]")
    h, c = dec_att.init_hidden(feats_sp[:1])
    emb_hist = None
    for t in range(3):
        tok = caps[:1, t:t+1]
        logits, (h, c), alpha, embed_cur = dec_att.step(tok, (h, c), feats_sp[:1], emb_hist)
        emb_hist = embed_cur if emb_hist is None else torch.cat([emb_hist, embed_cur], dim=1)
    print(f"  step() logits shape : {logits.shape}  alpha shape : {alpha.shape}")

    print(f"\nParams DecoderLSTM           : {dec.get_num_params():,}")
    print(f"Params DecoderWithAttention  : {dec_att.get_num_params():,}")

    print("\n" + "="*70)
    print("Tous les décoders fonctionnent !")
    print("="*70)