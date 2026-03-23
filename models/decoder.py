"""
Decoder LSTM pour Image Captioning
====================================

Améliorations v3 par rapport à v2 :

1. DecoderWithAttention.forward_with_alphas() : variante de forward() qui
   retourne également le tenseur des poids d'attention (B, T, P) à chaque
   pas de temps. Utilisé par le Trainer pour calculer la régularisation
   doubly stochastic qui corrige le biais vers les coins de la grille 7×7.

   Principe (Xu et al. 2015, §4.2.1) :
     Σ_t alpha[t, p] ≈ 1  pour tout p  →  le modèle couvre toute l'image
     Pénalité : λ · mean((1 - Σ_t alpha[:, :, p])²)

2. DecoderLSTM : beam search réel (inchangé depuis v2).

3. DecoderWithAttention : attention de Bahdanau complète (inchangée).
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

    def generate_beam_search(self, features, beam_width=5,
                             max_length=20, start_token=1, end_token=2):
        """Beam search réel."""
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

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return torch.tensor([best[1]], dtype=torch.long, device=device)

    def generate_diverse_beam_search(self, features, num_captions=5,
                                     beam_width=5, max_length=20,
                                     start_token=1, end_token=2,
                                     diversity_penalty=0.8):
        """
        Diverse Beam Search — génère num_captions captions différentes et pertinentes.

        Principe (Vijayakumar et al. 2016) :
          On divise les beam_width beams en num_captions groupes d'1 beam chacun.
          À chaque step, quand on sélectionne le token d'un groupe g, on soustrait
          une pénalité proportionnelle au nombre de fois où ce token a déjà été
          choisi par les groupes g-1, g-2, ... déjà traités à ce step.
          Résultat : les groupes sont poussés à explorer des tokens différents.

        Args:
            features         : (1, feature_dim) — UNE seule image
            num_captions     : nombre de captions distinctes à générer (défaut: 5)
            beam_width       : taille du faisceau par groupe (défaut: 5)
            max_length       : longueur max de chaque caption
            start_token      : index du token START
            end_token        : index du token END
            diversity_penalty: force de la pénalité (0 = beam search standard,
                               1 = pénalité forte, 2 = très diversifié mais
                               risque de perdre en qualité)

        Returns:
            list[str-indices] : liste de num_captions tensors (1, seq_len)
                                ordonnés du meilleur score au moins bon
        """
        device = features.device

        # ── Un faisceau indépendant par groupe ────────────────────────────────
        # Chaque groupe commence avec le même état caché initial
        hidden_init = self.init_hidden(features)

        # Structure : liste de num_captions faisceaux
        # Chaque faisceau = liste de (score, tokens, hidden)
        group_beams = [
            [(0.0, [start_token], hidden_init)]
            for _ in range(num_captions)
        ]
        group_completed = [[] for _ in range(num_captions)]

        for _ in range(max_length):
            # chosen_tokens[t] : liste des tokens choisis à ce step par les groupes 0..t-1
            # Sert à calculer la pénalité pour le groupe t
            chosen_tokens_so_far = []

            for g in range(num_captions):
                new_beams_g = []

                for score, tokens, h in group_beams[g]:
                    if tokens[-1] == end_token:
                        group_completed[g].append((score, tokens))
                        continue

                    inp = torch.tensor([[tokens[-1]]], dtype=torch.long,
                                       device=device)
                    emb             = self.embedding(inp)
                    lstm_out, h_new = self.lstm(emb, h)
                    log_probs       = F.log_softmax(
                        self.fc(lstm_out.squeeze(1)), dim=-1
                    )  # (1, vocab_size)

                    # ── Pénalité de diversité ─────────────────────────────────
                    # Pour chaque token déjà choisi par un groupe précédent à ce step,
                    # on soustrait diversity_penalty à son log_prob.
                    log_probs_div = log_probs.clone()
                    for prev_token in chosen_tokens_so_far:
                        log_probs_div[0, prev_token] -= diversity_penalty

                    topk_lp, topk_ids = log_probs_div.topk(beam_width, dim=-1)

                    for k in range(beam_width):
                        tok = topk_ids[0, k].item()
                        # Score réel = log_prob original (pas pénalisé) pour garder
                        # un classement final comparable entre groupes
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

                # Enregistrer les tokens de tête de ce groupe pour les groupes suivants
                if group_beams[g]:
                    head_token = group_beams[g][0][1][-1]  # dernier token du meilleur beam
                    chosen_tokens_so_far.append(head_token)

        # ── Collecter et sélectionner le meilleur de chaque groupe ────────────
        results = []
        for g in range(num_captions):
            # Ajouter les beams non terminés comme captions complètes
            for score, tokens, _ in group_beams[g]:
                group_completed[g].append((score, tokens))

            if group_completed[g]:
                best = max(group_completed[g],
                           key=lambda x: x[0] / max(len(x[1]), 1))
                results.append(best)

        # Trier par score décroissant et retourner les tensors
        results.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        return [
            torch.tensor([tokens], dtype=torch.long, device=device)
            for _, tokens in results
        ]

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
# DECODER AVEC ATTENTION VISUELLE
# ============================================================================

class DecoderWithAttention(nn.Module):
    """
    Decoder LSTM avec attention de Bahdanau.

    Nouveauté v3 :
      forward_with_alphas() — retourne (outputs, alphas) pour la régularisation
      doubly stochastic dans le Trainer.
    """

    def __init__(self, feature_dim, embedding_dim, hidden_dim,
                 vocab_size, attention_dim=256, dropout=0.5):
        super().__init__()

        self.feature_dim   = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.vocab_size    = vocab_size
        self.attention_dim = attention_dim   # ← sauvegardé pour load_model
        self.num_layers    = 1  # LSTMCell = 1 couche

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(feature_dim, hidden_dim, attention_dim)

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
            features : (B, num_pixels, feature_dim)
            captions : (B, seq_len)

        Returns:
            outputs  : (B, seq_len, vocab_size)
        """
        B, seq_len = captions.shape
        embeddings = self.dropout(self.embedding(captions))  # (B, T, emb_dim)
        h, c       = self.init_hidden(features)

        outputs = []
        for t in range(seq_len):
            context, _  = self.attention(features, h)
            lstm_input  = torch.cat([embeddings[:, t], context], dim=1)
            h, c        = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(self.dropout(h)).unsqueeze(1))

        return torch.cat(outputs, dim=1)                     # (B, T, vocab)

    def forward_with_alphas(self, features, captions):
        """
        Teacher forcing avec attention — retourne aussi les poids d'attention.

        Utilisé par le Trainer pour la régularisation doubly stochastic :
          pénalité = λ · mean((1 - Σ_t alpha[t, p])²)

        On veut que chaque région spatiale p soit regardée exactement une fois
        au total sur la séquence, ce qui force le modèle à distribuer son
        attention sur l'ensemble de la grille 7×7 plutôt que de toujours
        revenir aux mêmes coins.

        Args:
            features : (B, num_pixels, feature_dim)  — de EncoderSpatial
            captions : (B, seq_len)

        Returns:
            outputs : (B, seq_len, vocab_size)
            alphas  : (B, seq_len, num_pixels)       — poids d'attention
        """
        B, seq_len = captions.shape
        embeddings = self.dropout(self.embedding(captions))  # (B, T, emb_dim)
        h, c       = self.init_hidden(features)

        outputs     = []
        alphas_list = []

        for t in range(seq_len):
            context, alpha = self.attention(features, h)     # alpha : (B, P)
            lstm_input     = torch.cat([embeddings[:, t], context], dim=1)
            h, c           = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(self.dropout(h)).unsqueeze(1))
            alphas_list.append(alpha.unsqueeze(1))           # (B, 1, P)

        outputs = torch.cat(outputs, dim=1)                  # (B, T, vocab)
        alphas  = torch.cat(alphas_list, dim=1)              # (B, T, P)
        return outputs, alphas

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

    def generate_diverse_beam_search(self, features, num_captions=5,
                                     beam_width=5, max_length=20,
                                     start_token=1, end_token=2,
                                     diversity_penalty=0.8):
        """
        Diverse Beam Search avec attention — génère num_captions captions différentes.

        Principe (Vijayakumar et al. 2016) :
          On divise les beams en num_captions groupes indépendants.
          À chaque step, les groupes traitent leurs tokens dans l'ordre g=0,1,...
          Le groupe g reçoit une pénalité sur les tokens déjà choisis en tête
          par les groupes 0..g-1, ce qui le pousse à explorer d'autres mots.
          Le score final est calculé avec les log_probs réels (non pénalisés)
          pour garder un classement qualité comparable entre groupes.

        Args:
            features         : (1, num_pixels, feature_dim) — UNE seule image
            num_captions     : nombre de captions distinctes (défaut: 5)
            beam_width       : taille du faisceau par groupe (défaut: 5)
            max_length       : longueur max de chaque caption
            start_token      : index du token START
            end_token        : index du token END
            diversity_penalty: force de la pénalité (0 = standard, 0.8 = défaut,
                               2.0 = très diversifié)

        Returns:
            list[Tensor] : num_captions tensors (1, seq_len), du meilleur au moins bon
        """
        device = features.device
        h_init, c_init = self.init_hidden(features)

        # Un faisceau par groupe, tous initialisés avec le même état caché
        group_beams = [
            [(0.0, [start_token], h_init[0].clone(), c_init[0].clone())]
            for _ in range(num_captions)
        ]
        group_completed = [[] for _ in range(num_captions)]

        for _ in range(max_length):
            chosen_tokens_so_far = []  # tokens de tête des groupes déjà traités

            for g in range(num_captions):
                new_beams_g = []

                for score, tokens, bh, bc in group_beams[g]:
                    if tokens[-1] == end_token:
                        group_completed[g].append((score, tokens))
                        continue

                    inp = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                    emb = self.embedding(inp)
                    ctx, _ = self.attention(features, bh.unsqueeze(0))
                    bh_new, bc_new = self.lstm(
                        torch.cat([emb, ctx], dim=1),
                        (bh.unsqueeze(0), bc.unsqueeze(0))
                    )
                    bh_new = bh_new.squeeze(0)
                    bc_new = bc_new.squeeze(0)

                    log_probs = F.log_softmax(self.fc(bh_new), dim=-1)  # (vocab_size,)

                    # ── Pénalité de diversité ─────────────────────────────────
                    log_probs_div = log_probs.clone()
                    for prev_token in chosen_tokens_so_far:
                        log_probs_div[prev_token] -= diversity_penalty

                    topk_lp, topk_ids = log_probs_div.topk(beam_width)

                    for k in range(beam_width):
                        tok     = topk_ids[k].item()
                        real_lp = log_probs[tok].item()  # score réel, non pénalisé
                        new_beams_g.append((
                            score + real_lp,
                            tokens + [tok],
                            bh_new, bc_new
                        ))

                if not new_beams_g:
                    continue

                new_beams_g.sort(key=lambda x: x[0], reverse=True)
                group_beams[g] = new_beams_g[:beam_width]

                # Token de tête de ce groupe → pénalise les groupes suivants
                if group_beams[g]:
                    head_token = group_beams[g][0][1][-1]
                    chosen_tokens_so_far.append(head_token)

        # ── Sélectionner le meilleur beam de chaque groupe ────────────────────
        results = []
        for g in range(num_captions):
            for score, tokens, _, _ in group_beams[g]:
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
        """
        Greedy search qui retourne aussi les poids d'attention à chaque pas.
        """
        self.eval()
        device = features.device
        h, c   = self.init_hidden(features)
        inp    = torch.tensor([start_token], dtype=torch.long, device=device)

        tokens = []
        alphas = []

        with torch.no_grad():
            for _ in range(max_length):
                emb            = self.embedding(inp)
                context, alpha = self.attention(features, h)
                h, c           = self.lstm(
                    torch.cat([emb, context], dim=1), (h, c)
                )
                predicted = self.fc(h).argmax(dim=1)

                tokens.append(predicted.item())
                alphas.append(alpha.squeeze(0))

                if predicted.item() == end_token:
                    break
                inp = predicted

        alphas = torch.stack(alphas, dim=0)
        return tokens, alphas

    def generate_beam_search_with_attention(self, features, beam_width=5,
                                            max_length=20,
                                            start_token=1, end_token=2):
        """
        Beam search qui retourne aussi les poids d'attention de la meilleure
        hypothèse à chaque pas.
        """
        self.eval()
        device = features.device
        h, c   = self.init_hidden(features)

        beams     = [(0.0, [start_token], h[0], c[0], [])]
        completed = []

        with torch.no_grad():
            for _ in range(max_length):
                new_beams = []

                for score, tokens, bh, bc, beam_alphas in beams:
                    if tokens[-1] == end_token:
                        completed.append((score, tokens, beam_alphas))
                        continue

                    inp            = torch.tensor(
                        [tokens[-1]], dtype=torch.long, device=device
                    )
                    emb            = self.embedding(inp)
                    ctx, alpha     = self.attention(
                        features, bh.unsqueeze(0)
                    )
                    bh_new, bc_new = self.lstm(
                        torch.cat([emb, ctx], dim=1),
                        (bh.unsqueeze(0), bc.unsqueeze(0))
                    )
                    bh_new  = bh_new.squeeze(0)
                    bc_new  = bc_new.squeeze(0)
                    alpha_t = alpha.squeeze(0)

                    log_probs         = F.log_softmax(self.fc(bh_new), dim=-1)
                    topk_lp, topk_ids = log_probs.topk(beam_width)

                    for k in range(beam_width):
                        new_beams.append((
                            score + topk_lp[k].item(),
                            tokens + [topk_ids[k].item()],
                            bh_new, bc_new,
                            beam_alphas + [alpha_t]
                        ))

                if not new_beams:
                    break
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_width]

            for score, tokens, bh, bc, beam_alphas in beams:
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

    print("\n[DecoderWithAttention — forward + forward_with_alphas + beam]")
    dec_att  = DecoderWithAttention(feat_dim, emb_dim, hid_dim, vocab)
    feats_sp = torch.randn(B, 49, feat_dim)  # grille 7×7
    print(f"  forward()              : {dec_att(feats_sp, caps).shape}")
    outputs, alphas = dec_att.forward_with_alphas(feats_sp, caps)
    print(f"  forward_with_alphas()  : outputs={outputs.shape}, alphas={alphas.shape}")
    assert alphas.shape == (B, T, 49), f"Shape inattendue : {alphas.shape}"

    # Vérifier que la pénalité doubly stochastic est calculable
    attention_sum = alphas.sum(dim=1)              # (B, P)
    penalty = ((1.0 - attention_sum) ** 2).mean()
    print(f"  Pénalité doubly stoch  : {penalty.item():.6f}  (avant entraînement, valeur élevée attendue)")

    print(f"  Greedy        : {dec_att.generate(feats_sp[:1], max_length=8).shape}")
    print(f"  Beam (w=3)    : {dec_att.generate_beam_search(feats_sp[:1], beam_width=3, max_length=8).shape}")

    print(f"\nParams DecoderLSTM          : {dec.get_num_params():,}")
    print(f"Params DecoderWithAttention : {dec_att.get_num_params():,}")

    print("\n" + "="*70)
    print("Tous les decoders fonctionnent !")
    print("="*70)