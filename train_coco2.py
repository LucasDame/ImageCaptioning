"""
Script d'entraînement COCO pour Image Captioning
=================================================

Compatible avec les trois encoder_type du nouveau modèle :
  'lite'      → EncoderCNNLite  + DecoderLSTM
  'full'      → EncoderCNN      + DecoderLSTM  (résiduel)
  'attention' → EncoderSpatial  + DecoderWithAttention

Différences avec train.py (Flickr8k) :
  - Deux fichiers JSON séparés (train2017 / val2017) au lieu d'un split manuel
  - Import depuis preprocessing_coco et config_coco
  - attention_dim passé à create_model si encoder_type='attention'
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from utils import vocabulary, data_loader
from utils.preprocessing_coco import CaptionPreprocessor, ImagePreprocessor
from models2 import caption_model2

from config_coco2 import CONFIG


class Trainer:
    """
    Classe pour gérer l'entraînement du modèle.
    """

    def __init__(self, model, train_loader, val_loader, vocabulary, config):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.vocabulary   = vocabulary
        self.config       = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de : {self.device}")

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocabulary.word2idx[vocabulary.pad_token]
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.train_losses  = []
        self.val_losses    = []
        self.best_val_loss = float('inf')

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'],        exist_ok=True)

    # -------------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()
        total_loss  = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')

        for images, captions, lengths in pbar:
            images   = images.to(self.device)
            captions = captions.to(self.device)

            inputs  = captions[:, :-1]   # Tout sauf <END>
            targets = captions[:, 1:]    # Tout sauf <START>

            outputs = self.model(images, inputs)        # (B, T, vocab)
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)

            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    # -------------------------------------------------------------------------

    def validate(self):
        self.model.eval()
        total_loss  = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader,
                                                   desc='Validation'):
                images   = images.to(self.device)
                captions = captions.to(self.device)

                inputs  = captions[:, :-1]
                targets = captions[:, 1:]

                outputs = self.model(images, inputs)
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / num_batches

    # -------------------------------------------------------------------------

    def train(self):
        print("\n" + "="*70)
        print("DÉBUT DE L'ENTRAÎNEMENT (COCO)")
        print("="*70)

        params = self.model.get_num_params()
        print(f"\nNombre de paramètres : {params['total']:,}")
        print(f"  Encoder : {params['encoder']:,}")
        print(f"  Decoder : {params['decoder']:,}")

        print(f"\nConfiguration :")
        print(f"  Encoder type  : {self.config['encoder_type']}")
        print(f"  Epochs        : {self.config['num_epochs']}")
        print(f"  Batch size    : {self.config['batch_size']}")
        print(f"  Learning rate : {self.config['learning_rate']}")
        print(f"  Device        : {self.device}")

        start_time       = time.time()
        patience_counter = 0

        for epoch in range(self.config['num_epochs']):

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Val Loss   : {val_loss:.4f}")

            # Checkpoint régulier
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                ckpt_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                caption_model2.save_model(
                    self.model, ckpt_path,
                    optimizer=self.optimizer,
                    epoch=epoch, loss=val_loss,
                    vocab=self.vocabulary
                )

            # Meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config['checkpoint_dir'],
                                         'best_model.pth')
                caption_model2.save_model(
                    self.model, best_path,
                    optimizer=self.optimizer,
                    epoch=epoch, loss=val_loss,
                    vocab=self.vocabulary
                )
                patience_counter = 0
                print(f"  ✓ Nouveau meilleur modèle sauvegardé "
                      f"(loss : {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ✗ Pas d'amélioration "
                      f"(patience : {patience_counter}/{self.config['patience']})")

                if patience_counter >= self.config['patience']:
                    print("\nEarly stopping déclenché !")
                    break

        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/60:.2f} minutes")
        print(f"Meilleure validation loss : {self.best_val_loss:.4f}")

        self.plot_learning_curves()
        self.save_history()

    # -------------------------------------------------------------------------

    def plot_learning_curves(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss',      linewidth=2)
        plt.plot(epochs, self.val_losses,   'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss',  fontsize=12)
        plt.title(f'Learning Curves — COCO ({self.config["encoder_type"]})',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.config['log_dir'],
                                 'learning_curves_coco.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCourbes sauvegardées dans {save_path}")
        plt.close()

    def save_history(self):
        history = {
            'train_losses':  self.train_losses,
            'val_losses':    self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config':        self.config
        }
        save_path = os.path.join(self.config['log_dir'],
                                 'training_history_coco.json')
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Historique sauvegardé dans {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("PRÉPARATION DES DONNÉES COCO")
    print("="*70)

    # -------------------------------------------------------------------------
    # VOCABULAIRE
    # -------------------------------------------------------------------------

    if os.path.exists(CONFIG['vocab_path']):
        print(f"\nChargement du vocabulaire depuis {CONFIG['vocab_path']}")
        vocab = vocabulary.Vocabulary.load(CONFIG['vocab_path'])
    else:
        print("\nConstruction du vocabulaire depuis le train set COCO...")
        train_caption_prep = CaptionPreprocessor(
            CONFIG['train_captions_file'],
            CONFIG['train_images_dir']
        )
        vocab = vocabulary.Vocabulary(freq_threshold=CONFIG['freq_threshold'])
        vocab.build_vocabulary(train_caption_prep.get_all_captions())
        vocab.save(CONFIG['vocab_path'])

    print(f"Taille du vocabulaire : {len(vocab)}")

    # -------------------------------------------------------------------------
    # DONNÉES — SPLITS OFFICIELS COCO
    # -------------------------------------------------------------------------

    print("\nChargement des paires train (COCO train2017)...")
    train_caption_prep = CaptionPreprocessor(
        CONFIG['train_captions_file'],
        CONFIG['train_images_dir']
    )
    train_pairs = train_caption_prep.get_image_caption_pairs()

    print("\nChargement des paires val (COCO val2017)...")
    val_caption_prep = CaptionPreprocessor(
        CONFIG['val_captions_file'],
        CONFIG['val_images_dir']
    )
    val_pairs = val_caption_prep.get_image_caption_pairs()

    # -------------------------------------------------------------------------
    # DATALOADERS
    # -------------------------------------------------------------------------

    image_prep = ImagePreprocessor(
        image_size=CONFIG['image_size'], normalize=True
    )

    train_loader, val_loader = data_loader.get_data_loaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        vocabulary=vocab,
        image_preprocessor=image_prep,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        shuffle_train=True
    )

    # -------------------------------------------------------------------------
    # MODÈLE
    # -------------------------------------------------------------------------

    print(f"\nCréation du modèle (encoder_type='{CONFIG['encoder_type']}')...")
    model = caption_model2.create_model(
        vocab_size     = len(vocab),
        embedding_dim  = CONFIG['embedding_dim'],
        hidden_dim     = CONFIG['hidden_dim'],
        feature_dim    = CONFIG['feature_dim'],
        num_layers     = CONFIG['num_layers'],
        dropout        = CONFIG['dropout'],
        encoder_type   = CONFIG['encoder_type'],
        attention_dim  = CONFIG.get('attention_dim', 256),  # ignoré si pas 'attention'
    )

    # -------------------------------------------------------------------------
    # ENTRAÎNEMENT
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocab,
        config=CONFIG
    )

    trainer.train()

    print("\n" + "="*70)
    print("ENTRAÎNEMENT COCO TERMINÉ !")
    print("="*70)


if __name__ == "__main__":
    main()
