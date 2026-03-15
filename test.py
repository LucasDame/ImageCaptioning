"""
test.py — Suite de tests pour le projet Image Captioning COCO
=============================================================

Lance tous les tests sans avoir besoin des données COCO (tout est synthétique).

Utilisation :
    python test.py              # tous les tests
    python test.py -v           # verbeux (affiche chaque sous-test)
    python test.py TestEncoder  # un seul groupe
    python test.py -v TestTrainer  # un groupe en verbeux

Groupes de tests :
    TestConfig          — config.py : get_config, surcharges, clés obligatoires
    TestVocabulary      — vocabulary.py : construction, numericalize, save/load
    TestEncoder         — encoder.py : shapes de sortie, nombre de paramètres
    TestDecoder         — decoder.py : forward, forward_with_alphas, generate
    TestCaptionModel    — caption_model.py : create_model, save_model, load_model
    TestDataLoader      — data_loader.py : collate, padding, longueurs
    TestTrainer         — train.py : Trainer (plateau + cosine), 2 epochs synthétiques
    TestDemo            — demo.py : CaptionDemo (sur checkpoint synthétique)
    TestEvaluate        — evaluate.py : compute_cider_score, compute_bleu
    TestIntegration     — pipeline complet end-to-end (mini COCO synthétique)
"""

import os
import sys
import math
import tempfile
import unittest
import warnings

warnings.filterwarnings('ignore')

# ── Ajouter le projet au path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# =============================================================================
# UTILITAIRES PARTAGÉS
# =============================================================================

VOCAB_SIZE   = 200
EMBED_DIM    = 64
HIDDEN_DIM   = 128
FEATURE_DIM  = 128
ATTENTION_DIM = 64
BATCH_SIZE   = 4
SEQ_LEN      = 10
NUM_PIXELS   = 49   # grille 7×7


def make_images(B=BATCH_SIZE):
    """Batch de tenseurs image aléatoires (B, 3, 224, 224)."""
    return torch.randn(B, 3, 224, 224)


def make_captions(B=BATCH_SIZE, T=SEQ_LEN):
    """Batch de captions aléatoires (B, T) avec token 0=PAD."""
    return torch.randint(1, VOCAB_SIZE, (B, T))


def make_features_global(B=BATCH_SIZE):
    """Features vectorielles pour DecoderLSTM (B, feature_dim)."""
    return torch.randn(B, FEATURE_DIM)


def make_features_spatial(B=BATCH_SIZE):
    """Features spatiales pour DecoderWithAttention (B, 49, feature_dim)."""
    return torch.randn(B, NUM_PIXELS, FEATURE_DIM)


def make_tiny_vocab():
    """Crée un Vocabulary minimal avec 20 captions."""
    from utils.vocabulary import Vocabulary
    captions = [
        "a dog is running in the park",
        "two cats sitting on a wall",
        "a man riding a bicycle",
        "children playing in a garden",
        "a bird flying over the water",
        "a car parked on the street",
        "people walking in the city",
        "a woman reading a book",
        "a dog playing with a ball",
        "the cat is sleeping on the sofa",
        "a horse standing in a field",
        "two dogs running together",
        "a person jumping over a fence",
        "children eating ice cream",
        "a boat sailing on the sea",
        "a man cooking in a kitchen",
        "a woman smiling at the camera",
        "a cat looking out the window",
        "a dog swimming in a pool",
        "people sitting around a table",
    ]
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(captions)
    return vocab, captions


def save_fake_checkpoint(model, vocab, path):
    """Sauvegarde un checkpoint avec un modèle et un vocabulaire synthétiques."""
    from models.caption_model import save_model
    save_model(model, path, epoch=0, loss=3.0, vocab=vocab)


def make_fake_pil_image(path, size=(64, 64)):
    """Crée un fichier image PNG de couleur aléatoire."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


# =============================================================================
# TESTS CONFIG
# =============================================================================

class TestConfig(unittest.TestCase):
    """Teste config.py."""

    def test_get_config_all_models(self):
        from config import get_config
        for model in ['cnn', 'resnet', 'densenet']:
            cfg = get_config(model)
            self.assertIsInstance(cfg, dict)
            self.assertEqual(cfg['model'], model)

    def test_get_config_fast(self):
        from config import get_config
        cfg = get_config('densenet', fast=True)
        self.assertLessEqual(cfg['num_epochs'], 10)
        self.assertLessEqual(cfg['bleu_num_samples'], 500)

    def test_get_config_unknown_raises(self):
        from config import get_config
        with self.assertRaises(ValueError):
            get_config('unknown_model')

    def test_required_keys(self):
        from config import get_config
        required = [
            'model', 'train_captions_file', 'val_captions_file',
            'train_images_dir', 'val_images_dir', 'vocab_path',
            'checkpoint_dir', 'log_dir', 'embedding_dim', 'hidden_dim',
            'feature_dim', 'dropout', 'learning_rate', 'batch_size',
            'num_epochs', 'patience', 'cosine_T0', 'cosine_T_mult',
            'plateau_patience', 'plateau_factor', 'attention_lambda',
            'max_no_improve_cycles',
        ]
        for model in ['cnn', 'resnet', 'densenet']:
            cfg = get_config(model)
            for k in required:
                self.assertIn(k, cfg, f"Clé '{k}' absente de CONFIG_{model.upper()}")

    def test_cnn_attention_lambda_zero(self):
        from config import get_config
        cfg = get_config('cnn')
        self.assertEqual(cfg['attention_lambda'], 0.0,
                         "cnn ne doit pas utiliser la pénalité attention")

    def test_attention_models_lambda_positive(self):
        from config import get_config
        for model in ['resnet', 'densenet']:
            cfg = get_config(model)
            self.assertGreater(cfg['attention_lambda'], 0.0)

    def test_config_is_copy(self):
        """get_config doit retourner une copie — modifier une instance ne touche pas les autres."""
        from config import get_config
        cfg1 = get_config('densenet')
        cfg2 = get_config('densenet')
        cfg1['learning_rate'] = 999.0
        self.assertNotEqual(cfg2['learning_rate'], 999.0)

    def test_densenet_specific_keys(self):
        from config import get_config
        cfg = get_config('densenet')
        for k in ['growth_rate', 'compression', 'dense_dropout', 'block_config']:
            self.assertIn(k, cfg)
        self.assertEqual(cfg['growth_rate'], 32)
        self.assertEqual(len(cfg['block_config']), 4)


# =============================================================================
# TESTS VOCABULARY
# =============================================================================

class TestVocabulary(unittest.TestCase):
    """Teste utils/vocabulary.py."""

    def setUp(self):
        self.vocab, self.captions = make_tiny_vocab()

    def test_special_tokens_indices(self):
        v = self.vocab
        self.assertEqual(v.word2idx[v.pad_token],   0)
        self.assertEqual(v.word2idx[v.start_token], 1)
        self.assertEqual(v.word2idx[v.end_token],   2)
        self.assertEqual(v.word2idx[v.unk_token],   3)

    def test_vocab_size_positive(self):
        self.assertGreater(len(self.vocab), 10)

    def test_numericalize_starts_with_start(self):
        indices = self.vocab.numericalize("a dog is running")
        self.assertEqual(indices[0], self.vocab.word2idx[self.vocab.start_token])

    def test_numericalize_ends_with_end(self):
        indices = self.vocab.numericalize("a dog is running")
        self.assertEqual(indices[-1], self.vocab.word2idx[self.vocab.end_token])

    def test_denumericalize_roundtrip(self):
        text = "a dog is running"
        indices = self.vocab.numericalize(text)
        reconstructed = self.vocab.denumericalize(indices)
        self.assertEqual(reconstructed, text)

    def test_unknown_word_maps_to_unk(self):
        indices = self.vocab.numericalize("xyzzy_unknown_word")
        unk_idx = self.vocab.word2idx[self.vocab.unk_token]
        self.assertIn(unk_idx, indices)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'vocab.pkl')
            self.vocab.save(path)
            loaded = self.vocab.load(path)
            self.assertEqual(len(loaded), len(self.vocab))
            self.assertEqual(loaded.word2idx, self.vocab.word2idx)

    def test_denumericalize_stops_at_end(self):
        """Le texte reconstruit ne doit pas contenir END ou tokens après."""
        indices = self.vocab.numericalize("a cat sleeping")
        text = self.vocab.denumericalize(indices)
        self.assertNotIn('<END>', text)
        self.assertNotIn('<PAD>', text)

    def test_tensor_input_to_denumericalize(self):
        """Doit accepter un tensor PyTorch en entrée."""
        indices = self.vocab.numericalize("a dog running")
        tensor  = torch.tensor(indices)
        result  = self.vocab.denumericalize(tensor)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# =============================================================================
# TESTS ENCODER
# =============================================================================

class TestEncoder(unittest.TestCase):
    """Teste models/encoder.py — shapes et paramètres."""

    def _check_shape(self, encoder, expected_shape):
        images = make_images(B=2)
        out = encoder(images)
        self.assertEqual(out.shape, expected_shape,
                         f"{encoder.__class__.__name__}: shape {out.shape} ≠ {expected_shape}")

    def test_encoder_cnn_output_shape(self):
        from models.encoder import EncoderCNN
        enc = EncoderCNN(feature_dim=FEATURE_DIM)
        self._check_shape(enc, (2, FEATURE_DIM))

    def test_encoder_spatial_output_shape(self):
        from models.encoder import EncoderSpatial
        enc = EncoderSpatial(feature_dim=FEATURE_DIM, grid_size=7)
        self._check_shape(enc, (2, 49, FEATURE_DIM))

    def test_encoder_densenet_output_shape(self):
        from models.encoder import EncoderDenseNet
        enc = EncoderDenseNet(
            feature_dim=FEATURE_DIM, grid_size=7,
            growth_rate=8, compression=0.5,
            block_config=(2, 2, 2, 2)   # config allégée pour le test
        )
        self._check_shape(enc, (2, 49, FEATURE_DIM))

    def test_encoder_cnn_params_positive(self):
        from models.encoder import EncoderCNN
        enc = EncoderCNN(feature_dim=FEATURE_DIM)
        self.assertGreater(enc.get_num_params(), 0)

    def test_encoder_spatial_params_positive(self):
        from models.encoder import EncoderSpatial
        enc = EncoderSpatial(feature_dim=FEATURE_DIM)
        self.assertGreater(enc.get_num_params(), 0)

    def test_encoder_densenet_params_positive(self):
        from models.encoder import EncoderDenseNet
        enc = EncoderDenseNet(feature_dim=FEATURE_DIM, block_config=(2, 2, 2, 2), growth_rate=8)
        self.assertGreater(enc.get_num_params(), 0)

    def test_encoder_densenet_grid_size_respected(self):
        from models.encoder import EncoderDenseNet
        for grid in [5, 7, 14]:
            enc = EncoderDenseNet(feature_dim=64, grid_size=grid,
                                  block_config=(2, 2, 2, 2), growth_rate=8)
            out = enc(make_images(B=1))
            self.assertEqual(out.shape[1], grid * grid,
                             f"grid_size={grid} → attendu {grid*grid} pixels, obtenu {out.shape[1]}")

    def test_encoder_cnn_feature_dim_respected(self):
        from models.encoder import EncoderCNN
        for fdim in [64, 128, 256]:
            enc = EncoderCNN(feature_dim=fdim)
            out = enc(make_images(B=1))
            self.assertEqual(out.shape[-1], fdim)


# =============================================================================
# TESTS DECODER
# =============================================================================

class TestDecoder(unittest.TestCase):
    """Teste models/decoder.py."""

    def setUp(self):
        from models.decoder import DecoderLSTM, DecoderWithAttention
        self.lstm = DecoderLSTM(
            feature_dim=FEATURE_DIM, embedding_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, dropout=0.0
        )
        self.att = DecoderWithAttention(
            feature_dim=FEATURE_DIM, embedding_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
            attention_dim=ATTENTION_DIM, dropout=0.0
        )

    def test_lstm_forward_shape(self):
        feats = make_features_global()
        caps  = make_captions()
        out   = self.lstm(feats, caps)
        self.assertEqual(out.shape, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))

    def test_att_forward_shape(self):
        feats = make_features_spatial()
        caps  = make_captions()
        out   = self.att(feats, caps)
        self.assertEqual(out.shape, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))

    def test_forward_with_alphas_shapes(self):
        feats       = make_features_spatial()
        caps        = make_captions()
        out, alphas = self.att.forward_with_alphas(feats, caps)
        self.assertEqual(out.shape,    (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        self.assertEqual(alphas.shape, (BATCH_SIZE, SEQ_LEN, NUM_PIXELS))

    def test_alphas_sum_to_one_per_timestep(self):
        """Les poids d'attention doivent sommer à 1 pour chaque timestep."""
        feats       = make_features_spatial(B=2)
        caps        = make_captions(B=2, T=5)
        _, alphas   = self.att.forward_with_alphas(feats, caps)
        sums = alphas.sum(dim=2)  # (B, T)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
                        f"Somme des alphas ≠ 1 : {sums}")

    def test_lstm_greedy_generate_shape(self):
        feats     = make_features_global(B=1)
        generated = self.lstm.generate(feats, max_length=8, start_token=1, end_token=2)
        self.assertEqual(generated.shape[0], 1)
        self.assertLessEqual(generated.shape[1], 8)

    def test_att_greedy_generate_shape(self):
        feats     = make_features_spatial(B=1)
        generated = self.att.generate(feats, max_length=8, start_token=1, end_token=2)
        self.assertEqual(generated.shape[0], 1)
        self.assertLessEqual(generated.shape[1], 8)

    def test_lstm_beam_search_shape(self):
        feats     = make_features_global(B=1)
        generated = self.lstm.generate_beam_search(feats, beam_width=3,
                                                   max_length=8,
                                                   start_token=1, end_token=2)
        self.assertEqual(generated.shape[0], 1)

    def test_att_beam_search_shape(self):
        feats     = make_features_spatial(B=1)
        generated = self.att.generate_beam_search(feats, beam_width=3,
                                                  max_length=8,
                                                  start_token=1, end_token=2)
        self.assertEqual(generated.shape[0], 1)

    def test_att_generate_with_attention_shapes(self):
        feats          = make_features_spatial(B=1)
        tokens, alphas = self.att.generate_with_attention(
            feats, max_length=6, start_token=1, end_token=2
        )
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(alphas, torch.Tensor)
        # alphas : (T, num_pixels)
        self.assertEqual(alphas.shape[1], NUM_PIXELS)

    def test_att_beam_with_attention_shapes(self):
        feats          = make_features_spatial(B=1)
        tokens, alphas = self.att.generate_beam_search_with_attention(
            feats, beam_width=3, max_length=6, start_token=1, end_token=2
        )
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(alphas, torch.Tensor)

    def test_decoder_params_positive(self):
        self.assertGreater(self.lstm.get_num_params(), 0)
        self.assertGreater(self.att.get_num_params(), 0)


# =============================================================================
# TESTS CAPTION MODEL
# =============================================================================

class TestCaptionModel(unittest.TestCase):
    """Teste models/caption_model.py."""

    def _make_model(self, arch):
        from models.caption_model import create_model
        kwargs = {}
        if arch == 'densenet':
            kwargs = {'growth_rate': 8, 'block_config': (2, 2, 2, 2)}
        return create_model(
            vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, feature_dim=FEATURE_DIM,
            attention_dim=ATTENTION_DIM, dropout=0.0,
            model=arch, **kwargs
        )

    def test_create_all_architectures(self):
        for arch in ['cnn', 'resnet', 'densenet']:
            with self.subTest(arch=arch):
                m = self._make_model(arch)
                self.assertIsNotNone(m)

    def test_forward_all_architectures(self):
        images = make_images(B=2)
        caps   = make_captions(B=2, T=SEQ_LEN)
        for arch in ['cnn', 'resnet', 'densenet']:
            with self.subTest(arch=arch):
                m   = self._make_model(arch)
                out = m(images, caps)
                self.assertEqual(out.shape, (2, SEQ_LEN, VOCAB_SIZE))

    def test_forward_with_alphas_attention_models(self):
        images = make_images(B=2)
        caps   = make_captions(B=2, T=SEQ_LEN)
        for arch in ['resnet', 'densenet']:
            with self.subTest(arch=arch):
                m           = self._make_model(arch)
                out, alphas = m.forward_with_alphas(images, caps)
                self.assertEqual(out.shape,    (2, SEQ_LEN, VOCAB_SIZE))
                self.assertEqual(alphas.shape, (2, SEQ_LEN, NUM_PIXELS))

    def test_forward_with_alphas_raises_for_cnn(self):
        m = self._make_model('cnn')
        images = make_images(B=2)
        caps   = make_captions(B=2, T=SEQ_LEN)
        with self.assertRaises(ValueError):
            m.forward_with_alphas(images, caps)

    def test_get_num_params(self):
        for arch in ['cnn', 'resnet', 'densenet']:
            m = self._make_model(arch)
            p = m.get_num_params()
            self.assertIn('encoder', p)
            self.assertIn('decoder', p)
            self.assertIn('total', p)
            self.assertEqual(p['total'], p['encoder'] + p['decoder'])

    def test_generate_caption_greedy(self):
        for arch in ['cnn', 'resnet', 'densenet']:
            with self.subTest(arch=arch):
                m     = self._make_model(arch)
                image = make_images(B=1)
                m.eval()
                cap = m.generate_caption(image, max_length=8,
                                         start_token=1, end_token=2,
                                         method='greedy')
                self.assertIsInstance(cap, torch.Tensor)

    def test_save_and_load_roundtrip(self):
        from models.caption_model import save_model, load_model
        vocab, _ = make_tiny_vocab()

        for arch in ['cnn', 'resnet', 'densenet']:
            with self.subTest(arch=arch):
                m1 = self._make_model(arch)
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, f'model_{arch}.pth')
                    save_model(m1, path, epoch=5, loss=2.5, vocab=vocab)
                    self.assertTrue(os.path.exists(path))

                    m2, info = load_model(path, device='cpu')
                    self.assertEqual(info['epoch'], 5)
                    self.assertAlmostEqual(info['loss'],  2.5, places=4)
                    self.assertIsNotNone(info['vocab'])

                    # Les poids doivent être identiques
                    for p1, p2 in zip(m1.parameters(), m2.parameters()):
                        self.assertTrue(torch.allclose(p1, p2))

    def test_load_auto_detects_arch(self):
        from models.caption_model import save_model, load_model
        vocab, _ = make_tiny_vocab()
        m1 = self._make_model('densenet')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'densenet.pth')
            save_model(m1, path, vocab=vocab)
            m2, info = load_model(path, device='cpu')
            from models.encoder import EncoderDenseNet
            self.assertIsInstance(m2.encoder, EncoderDenseNet)

    def test_unknown_arch_raises(self):
        from models.caption_model import create_model
        with self.assertRaises(ValueError):
            create_model(vocab_size=100, model='invalid_arch')


# =============================================================================
# TESTS DATA LOADER
# =============================================================================

class TestDataLoader(unittest.TestCase):
    """Teste utils/data_loader.py."""

    def _make_fake_pairs(self, n=20):
        """Crée des paires (image_path, caption) avec des vraies images PNG."""
        tmpdir = tempfile.mkdtemp()
        pairs  = []
        for i in range(n):
            img_path = os.path.join(tmpdir, f'img_{i:03d}.png')
            make_fake_pil_image(img_path, size=(64, 64))
            pairs.append({
                'image_path': img_path,
                'caption':    f'a dog is running in the park sample {i}',
                'image_name': f'img_{i:03d}.png',
            })
        return pairs, tmpdir

    def setUp(self):
        self.vocab, _ = make_tiny_vocab()
        self.pairs, self.tmpdir = self._make_fake_pairs(20)

        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        from utils.preprocessing import ImagePreprocessor
        self.image_prep = ImagePreprocessor(
            image_size=32, normalize=False,
            train_transform=val_transform,
            val_transform=val_transform,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dataset_length(self):
        from utils.data_loader import ImageCaptionDataset
        ds = ImageCaptionDataset(self.pairs, self.vocab, self.image_prep)
        self.assertEqual(len(ds), len(self.pairs))

    def test_dataset_getitem_shapes(self):
        from utils.data_loader import ImageCaptionDataset
        ds      = ImageCaptionDataset(self.pairs, self.vocab, self.image_prep)
        img, cap = ds[0]
        self.assertEqual(img.shape[0], 3)     # canaux RGB
        self.assertIsInstance(cap, torch.Tensor)
        self.assertGreater(len(cap), 2)        # START + au moins un mot + END

    def test_collate_padding(self):
        from utils.data_loader import ImageCaptionDataset, CaptionCollate
        ds      = ImageCaptionDataset(self.pairs, self.vocab, self.image_prep)
        pad_idx = self.vocab.word2idx[self.vocab.pad_token]
        collate = CaptionCollate(pad_idx=pad_idx)
        loader  = DataLoader(ds, batch_size=4, collate_fn=collate)
        images, captions, lengths = next(iter(loader))
        # Toutes les captions du batch ont la même longueur après padding
        self.assertEqual(captions.shape[0], 4)
        self.assertEqual(captions.shape[1], lengths.max().item())

    def test_lengths_match_captions(self):
        from utils.data_loader import ImageCaptionDataset, CaptionCollate
        ds      = ImageCaptionDataset(self.pairs, self.vocab, self.image_prep)
        pad_idx = self.vocab.word2idx[self.vocab.pad_token]
        collate = CaptionCollate(pad_idx=pad_idx)
        loader  = DataLoader(ds, batch_size=8, collate_fn=collate)
        images, captions, lengths = next(iter(loader))
        for i, length in enumerate(lengths):
            # Les tokens après 'length' doivent être du padding
            if length < captions.shape[1]:
                self.assertTrue(
                    (captions[i, length:] == pad_idx).all(),
                    f"Token non-PAD après longueur réelle pour sample {i}"
                )

    def test_get_data_loaders(self):
        from utils.data_loader import get_data_loaders
        train_l, val_l = get_data_loaders(
            train_pairs=self.pairs[:16],
            val_pairs=self.pairs[16:],
            vocabulary=self.vocab,
            image_preprocessor=self.image_prep,
            batch_size=4, num_workers=0, shuffle_train=False
        )
        self.assertEqual(len(train_l.dataset), 16)
        self.assertEqual(len(val_l.dataset),   4)


# =============================================================================
# TESTS TRAINER (train.py)
# =============================================================================

class TestTrainer(unittest.TestCase):
    """Teste la boucle d'entraînement sur 2 epochs avec données synthétiques."""

    def _make_loader(self, pairs, vocab, image_prep):
        from utils.data_loader import get_data_loaders
        train_l, val_l = get_data_loaders(
            train_pairs=pairs[:16], val_pairs=pairs[16:],
            vocabulary=vocab, image_preprocessor=image_prep,
            batch_size=4, num_workers=0, shuffle_train=False
        )
        return train_l, val_l

    def _make_pairs_and_prep(self, n=20):
        tmpdir = tempfile.mkdtemp()
        pairs  = []
        for i in range(n):
            img_path = os.path.join(tmpdir, f'img_{i:03d}.png')
            make_fake_pil_image(img_path, size=(64, 64))
            pairs.append({'image_path': img_path,
                          'caption': f'a dog running sample {i}',
                          'image_name': f'img_{i:03d}.png'})
        t = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        ])
        from utils.preprocessing import ImagePreprocessor
        prep = ImagePreprocessor(image_size=32, normalize=False,
                                 train_transform=t, val_transform=t)
        return pairs, prep, tmpdir

    def _make_mini_config(self, arch, scheduler, checkpoint_dir, log_dir):
        from config import get_config
        cfg = get_config(arch, fast=True)
        cfg['model']          = arch
        cfg['num_epochs']     = 2
        cfg['batch_size']     = 4
        cfg['bleu_every']     = 999     # désactive les métriques lentes
        cfg['bleu_num_samples'] = 2
        cfg['checkpoint_dir'] = checkpoint_dir
        cfg['log_dir']        = log_dir
        cfg['warmup_epochs']  = 1
        cfg['patience']       = 10
        cfg['cosine_T0']      = 2
        cfg['save_every']     = 10
        cfg['attention_lambda'] = 0.0   # plus rapide
        if arch == 'densenet':
            cfg['growth_rate']  = 8
            cfg['block_config'] = (2, 2, 2, 2)
        return cfg

    def _run_trainer(self, arch, scheduler_type):
        from models.caption_model import create_model
        from train import Trainer

        vocab, _ = make_tiny_vocab()
        pairs, prep, tmpdir = self._make_pairs_and_prep(20)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            log_dir = os.path.join(ckpt_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)

            cfg = self._make_mini_config(arch, scheduler_type, ckpt_dir, log_dir)

            kwargs = {}
            if arch == 'densenet':
                kwargs = {'growth_rate': 8, 'block_config': (2, 2, 2, 2)}
            model = create_model(
                vocab_size=len(vocab), embedding_dim=EMBED_DIM,
                hidden_dim=HIDDEN_DIM, feature_dim=FEATURE_DIM,
                attention_dim=ATTENTION_DIM, dropout=0.0,
                model=arch, **kwargs
            )

            t = transforms.Compose([
                transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
            ])
            from utils.data_loader import get_data_loaders
            from utils.preprocessing import ImagePreprocessor
            prep = ImagePreprocessor(image_size=32, normalize=False,
                                     train_transform=t, val_transform=t)
            train_l, val_l = get_data_loaders(
                train_pairs=pairs[:16], val_pairs=pairs[16:],
                vocabulary=vocab, image_preprocessor=prep,
                batch_size=4, num_workers=0, shuffle_train=False
            )

            trainer = Trainer(
                model=model, train_loader=train_l, val_loader=val_l,
                vocabulary=vocab, config=cfg,
                scheduler_type=scheduler_type,
                val_pairs=pairs[16:]
            )
            trainer.train()

            # Vérifications post-entraînement
            self.assertEqual(len(trainer.train_losses), cfg['num_epochs'])
            self.assertEqual(len(trainer.val_losses),   cfg['num_epochs'])
            self.assertLess(trainer.best_val_loss, float('inf'))

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_trainer_cnn_plateau(self):
        self._run_trainer('cnn', 'plateau')

    def test_trainer_resnet_cosine(self):
        self._run_trainer('resnet', 'cosine')

    def test_trainer_densenet_cosine(self):
        self._run_trainer('densenet', 'cosine')

    def test_trainer_densenet_plateau(self):
        self._run_trainer('densenet', 'plateau')

    def test_trainer_invalid_scheduler(self):
        from train import Trainer
        from models.caption_model import create_model
        vocab, _ = make_tiny_vocab()
        model = create_model(vocab_size=len(vocab), model='cnn',
                             embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                             feature_dim=FEATURE_DIM, dropout=0.0)
        cfg = {'model': 'cnn', 'learning_rate': 1e-3, 'weight_decay': 0,
               'plateau_patience': 5, 'plateau_factor': 0.5, 'lr_min': 1e-6,
               'cosine_T0': 5, 'cosine_T_mult': 2, 'checkpoint_dir': '/tmp',
               'log_dir': '/tmp', 'num_epochs': 1, 'attention_lambda': 0.0}

        # Besoin d'un vrai loader pour le __init__, on passe juste l'init
        with self.assertRaises(ValueError):
            # Le ValueError est levé dans __init__ avant tout entraînement
            t_fake = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
            from utils.data_loader import get_data_loaders
            from utils.preprocessing import ImagePreprocessor
            tmpdir = tempfile.mkdtemp()
            pairs = []
            for i in range(8):
                p = os.path.join(tmpdir, f'img_{i}.png')
                make_fake_pil_image(p, (32,32))
                pairs.append({'image_path':p,'caption':'a dog','image_name':f'img_{i}.png'})
            prep = ImagePreprocessor(32, False, t_fake, t_fake)
            tl, vl = get_data_loaders(pairs[:4], pairs[4:], vocab, prep, 4, 0)
            import shutil
            Trainer(model, tl, vl, vocab, cfg, scheduler_type='invalid_scheduler')
            shutil.rmtree(tmpdir)


# =============================================================================
# TESTS DEMO
# =============================================================================

class TestDemo(unittest.TestCase):
    """Teste demo.py sur un checkpoint synthétique."""

    def setUp(self):
        from models.caption_model import create_model
        self.vocab, _ = make_tiny_vocab()
        self.tmpdir   = tempfile.mkdtemp()

        self.model = create_model(
            vocab_size=len(self.vocab), embedding_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, feature_dim=FEATURE_DIM,
            attention_dim=ATTENTION_DIM, dropout=0.0, model='densenet',
            growth_rate=8, block_config=(2, 2, 2, 2)
        )

        self.ckpt_path = os.path.join(self.tmpdir, 'model.pth')
        save_fake_checkpoint(self.model, self.vocab, self.ckpt_path)

        self.img_path = os.path.join(self.tmpdir, 'test.png')
        make_fake_pil_image(self.img_path, (128, 128))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_demo(self):
        from demo import CaptionDemo
        t = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        ])
        demo = CaptionDemo(checkpoint_path=self.ckpt_path)
        # Remplacer le préprocesseur par une version légère pour le test
        from utils.preprocessing import ImagePreprocessor
        demo.image_prep = ImagePreprocessor(32, False, t, t)
        return demo

    def test_generate_caption_returns_string(self):
        demo = self._make_demo()
        cap  = demo.generate_caption(self.img_path, method='greedy', max_length=8)
        self.assertIsInstance(cap, str)

    def test_generate_caption_beam_search(self):
        demo = self._make_demo()
        cap  = demo.generate_caption(self.img_path, method='beam_search',
                                     beam_width=3, max_length=8)
        self.assertIsInstance(cap, str)

    def test_generate_caption_file_not_found(self):
        demo = self._make_demo()
        with self.assertRaises(FileNotFoundError):
            demo.generate_caption('/nonexistent/path/image.jpg')

    def test_demo_single_saves_figure(self):
        demo     = self._make_demo()
        save_dir = os.path.join(self.tmpdir, 'out')
        demo.demo_single(self.img_path, method='greedy', max_length=6,
                         save_dir=save_dir)
        # Au moins un fichier PNG doit avoir été créé
        pngs = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        self.assertGreater(len(pngs), 0)

    def test_demo_folder_returns_dict(self):
        demo    = self._make_demo()
        out_dir = os.path.join(self.tmpdir, 'folder_out')
        results = demo.demo_folder(self.tmpdir, method='greedy',
                                   max_length=6, save_dir=out_dir)
        # Au moins l'image de test doit avoir été traitée
        png_names = [k for k in results.keys() if k.endswith('.png')]
        self.assertGreater(len(png_names), 0)


# =============================================================================
# TESTS EVALUATE
# =============================================================================

class TestEvaluate(unittest.TestCase):
    """Teste les fonctions de calcul de métriques dans evaluate.py."""

    def test_cider_perfect_match(self):
        """Avec un corpus d'au moins 2 documents, un match parfait doit scorer > 0.
        (Avec 1 seul document, IDF = log(2/2) = 0 pour tous les n-grammes → score nul.)"""
        from evaluate import compute_cider_score
        gen  = [['a', 'dog', 'running'], ['a', 'cat', 'sitting']]
        refs = [
            [['a', 'dog', 'running'], ['a', 'dog', 'running']],
            [['a', 'cat', 'sitting'], ['a', 'cat', 'sitting']],
        ]
        score = compute_cider_score(gen, refs)
        self.assertGreater(score, 0)

    def test_cider_no_overlap_low_score(self):
        from evaluate import compute_cider_score
        gen  = [['xyzzy', 'foobar', 'qux']]
        refs = [[['a', 'dog', 'running'], ['cat', 'sitting', 'wall']]]
        score = compute_cider_score(gen, refs)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_cider_multiple_samples(self):
        from evaluate import compute_cider_score
        gen  = [['a', 'cat'], ['a', 'dog']]
        refs = [[['a', 'cat', 'sleeping']], [['a', 'dog', 'running']]]
        score = compute_cider_score(gen, refs)
        self.assertGreaterEqual(score, 0.0)

    def test_cider_empty_returns_zero(self):
        from evaluate import compute_cider_score
        score = compute_cider_score([], [])
        self.assertEqual(score, 0.0)

    def test_bleu_available_or_skipped(self):
        """BLEU doit soit fonctionner, soit retourner None gracieusement."""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            gen  = [['a', 'dog', 'running']]
            refs = [[['a', 'dog', 'running']]]
            smooth = SmoothingFunction().method1
            score = corpus_bleu(refs, gen, weights=(1,0,0,0), smoothing_function=smooth)
            self.assertGreater(score, 0)
        except ImportError:
            self.skipTest("NLTK non installé — BLEU ignoré")

    def test_compute_cider_train_vs_evaluate_consistent(self):
        """Les deux implémentations CIDEr (train.py et evaluate.py) doivent donner le même résultat."""
        from train import compute_cider_score as cider_train
        from evaluate import compute_cider_score as cider_eval
        # 3 documents pour avoir un IDF non-trivial
        gen  = [['a', 'dog', 'running'], ['a', 'cat', 'sitting'], ['a', 'bird', 'flying']]
        refs = [
            [['a', 'dog', 'running', 'fast']],
            [['a', 'cat', 'sitting', 'still']],
            [['a', 'bird', 'flying', 'high']],
        ]
        self.assertAlmostEqual(
            cider_train(gen, refs),
            cider_eval(gen, refs),
            places=8,
            msg="Les deux implémentations CIDEr divergent"
        )


# =============================================================================
# TESTS VISUALIZE ATTENTION
# =============================================================================

class TestVisualizeAttention(unittest.TestCase):
    """Teste visualize_attention.py sur un checkpoint synthétique."""

    def setUp(self):
        from models.caption_model import create_model
        self.vocab, _ = make_tiny_vocab()
        self.tmpdir   = tempfile.mkdtemp()

        self.model = create_model(
            vocab_size=len(self.vocab), embedding_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, feature_dim=FEATURE_DIM,
            attention_dim=ATTENTION_DIM, dropout=0.0, model='densenet',
            growth_rate=8, block_config=(2, 2, 2, 2)
        )
        self.ckpt_path = os.path.join(self.tmpdir, 'model.pth')
        save_fake_checkpoint(self.model, self.vocab, self.ckpt_path)

        self.img_path = os.path.join(self.tmpdir, 'test.png')
        make_fake_pil_image(self.img_path, (128, 128))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_visualizer(self):
        from visualize_attention import AttentionVisualizer
        t = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        ])
        vis = AttentionVisualizer(checkpoint_path=self.ckpt_path)
        from utils.preprocessing import ImagePreprocessor
        vis.image_prep = ImagePreprocessor(32, False, t, t)
        return vis

    def test_get_caption_and_attention_shapes(self):
        vis          = self._make_visualizer()
        words, alphas = vis.get_caption_and_attention(
            self.img_path, method='greedy', max_length=6
        )
        self.assertIsInstance(words, list)
        self.assertIsInstance(alphas, torch.Tensor)
        # alphas : (T, num_pixels)
        self.assertEqual(alphas.shape[1], vis.grid_size ** 2)

    def test_visualize_saves_figure(self):
        vis     = self._make_visualizer()
        out_dir = os.path.join(self.tmpdir, 'attn_out')
        result  = vis.visualize(self.img_path, method='greedy', max_length=6,
                                save_dir=out_dir)
        # Le dossier doit toujours être créé
        self.assertTrue(os.path.exists(out_dir), "Le dossier de sortie n'a pas été créé")
        # Si la caption contient des mots visualisables, un PNG doit exister
        if result is not None:
            pngs = [f for f in os.listdir(out_dir) if f.endswith('.png')]
            self.assertGreater(len(pngs), 0, "Aucun PNG généré malgré une caption non vide")

    def test_cnn_model_raises(self):
        """CNN ne supporte pas l'attention — doit lever ValueError à l'init."""
        from models.caption_model import create_model, save_model
        from visualize_attention import AttentionVisualizer
        cnn_model = create_model(
            vocab_size=len(self.vocab), model='cnn',
            embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
            feature_dim=FEATURE_DIM, dropout=0.0
        )
        cnn_path = os.path.join(self.tmpdir, 'cnn_model.pth')
        save_model(cnn_model, cnn_path, vocab=self.vocab)
        with self.assertRaises(ValueError):
            AttentionVisualizer(checkpoint_path=cnn_path)


# =============================================================================
# TEST D'INTÉGRATION END-TO-END
# =============================================================================

class TestIntegration(unittest.TestCase):
    """
    Pipeline complet end-to-end :
      1. Créer un mini-dataset synthétique
      2. Entraîner 2 epochs (densenet + cosine)
      3. Charger le meilleur checkpoint
      4. Générer une caption
      5. Visualiser l'attention
      6. Évaluer CIDEr
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_pipeline(self):
        from utils.vocabulary import Vocabulary
        from utils.preprocessing import ImagePreprocessor
        from utils.data_loader import get_data_loaders
        from models.caption_model import create_model, load_model
        from train import Trainer
        from demo import CaptionDemo
        from visualize_attention import AttentionVisualizer
        from evaluate import compute_cider_score

        # ── Données synthétiques ──────────────────────────────────────────────
        captions_text = [
            "a dog running in the park",
            "a cat sitting on the wall",
            "two dogs playing together",
            "a bird flying in the sky",
            "a man riding a bicycle",
            "children playing in the garden",
            "a woman reading a book",
            "a cat sleeping on the sofa",
            "a horse standing in a field",
            "a boat sailing on the sea",
        ] * 4   # 40 paires au total

        pairs = []
        for i, cap in enumerate(captions_text):
            img_path = os.path.join(self.tmpdir, f'img_{i:03d}.png')
            make_fake_pil_image(img_path, (64, 64))
            pairs.append({'image_path': img_path, 'caption': cap,
                          'image_name': f'img_{i:03d}.png'})

        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary(captions_text)

        t = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        ])
        prep = ImagePreprocessor(32, False, t, t)

        train_l, val_l = get_data_loaders(
            train_pairs=pairs[:32], val_pairs=pairs[32:],
            vocabulary=vocab, image_preprocessor=prep,
            batch_size=4, num_workers=0, shuffle_train=False
        )

        # ── Entraînement (2 epochs, densenet allégé) ──────────────────────────
        ckpt_dir = os.path.join(self.tmpdir, 'ckpt')
        log_dir  = os.path.join(self.tmpdir, 'logs')
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir,  exist_ok=True)

        model = create_model(
            vocab_size=len(vocab), embedding_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, feature_dim=FEATURE_DIM,
            attention_dim=ATTENTION_DIM, dropout=0.0,
            model='densenet', growth_rate=8, block_config=(2, 2, 2, 2)
        )

        cfg = {
            'model': 'densenet', 'num_epochs': 2, 'batch_size': 4,
            'learning_rate': 1e-3, 'weight_decay': 0,
            'warmup_epochs': 1, 'patience': 10,
            'cosine_T0': 2, 'cosine_T_mult': 2, 'lr_min': 1e-6,
            'plateau_patience': 5, 'plateau_factor': 0.5,
            'attention_lambda': 1.0, 'max_no_improve_cycles': 3,
            'bleu_every': 999, 'bleu_num_samples': 2,
            'max_caption_length': 8, 'save_every': 10,
            'checkpoint_dir': ckpt_dir, 'log_dir': log_dir,
        }

        trainer = Trainer(
            model=model, train_loader=train_l, val_loader=val_l,
            vocabulary=vocab, config=cfg,
            scheduler_type='cosine', val_pairs=pairs[32:]
        )
        trainer.train()

        # ── Le checkpoint best_model.pth doit exister ─────────────────────────
        best_ckpt = os.path.join(ckpt_dir, 'best_model.pth')
        self.assertTrue(os.path.exists(best_ckpt),
                        "best_model.pth introuvable après l'entraînement")

        # ── Chargement et génération ──────────────────────────────────────────
        m2, info = load_model(best_ckpt, device='cpu')
        self.assertIsNotNone(info['vocab'])

        demo = CaptionDemo(checkpoint_path=best_ckpt)
        demo.image_prep = ImagePreprocessor(32, False, t, t)

        cap = demo.generate_caption(pairs[0]['image_path'],
                                    method='greedy', max_length=8)
        self.assertIsInstance(cap, str)

        # ── Visualisation de l'attention ──────────────────────────────────────
        vis = AttentionVisualizer(checkpoint_path=best_ckpt)
        vis.image_prep = ImagePreprocessor(32, False, t, t)

        attn_dir = os.path.join(self.tmpdir, 'attn')
        vis.visualize(pairs[0]['image_path'], method='greedy',
                      max_length=6, save_dir=attn_dir)
        # Le dossier doit toujours être créé (même si caption = que des stop words)
        self.assertTrue(os.path.exists(attn_dir), "Le dossier attention n'a pas été créé")

        # ── Évaluation CIDEr ──────────────────────────────────────────────────
        gen  = [['a', 'dog', 'running'], ['a', 'cat', 'sitting']]
        refs = [[['a', 'dog', 'running', 'park']], [['a', 'cat', 'sitting', 'wall']]]
        score = compute_cider_score(gen, refs)
        self.assertGreaterEqual(score, 0.0)

        # ── Historique sauvegardé ─────────────────────────────────────────────
        self.assertEqual(len(trainer.train_losses), 2)
        self.assertLess(trainer.best_val_loss, float('inf'))


# =============================================================================
# RUNNER
# =============================================================================

def main():
    # Afficher l'en-tête
    print("=" * 70)
    print("TESTS — Image Captioning COCO")
    print("=" * 70)
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {'disponible' if torch.cuda.is_available() else 'non disponible'}")
    print("=" * 70)
    print()

    # Lancer avec unittest
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestConfig,
        TestVocabulary,
        TestEncoder,
        TestDecoder,
        TestCaptionModel,
        TestDataLoader,
        TestTrainer,
        TestDemo,
        TestEvaluate,
        TestVisualizeAttention,
        TestIntegration,
    ]

    # Si des arguments sont passés (ex: python test.py TestEncoder), filtrer
    args = sys.argv[1:]
    filter_args = [a for a in args if not a.startswith('-')]

    for cls in test_classes:
        if not filter_args or cls.__name__ in filter_args:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    # Verbosité
    verbosity = 2 if '-v' in args else 1

    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print(f"✓ Tous les tests passent ({result.testsRun} tests)")
    else:
        n_fail = len(result.failures) + len(result.errors)
        print(f"✗ {n_fail} test(s) en échec sur {result.testsRun}")
        sys.exit(1)


if __name__ == '__main__':
    main()