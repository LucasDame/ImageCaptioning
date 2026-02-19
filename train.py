"""
Script d'entraînement pour Image Captioning
============================================

Entraîne le modèle encoder-decoder sur le dataset Flickr8k
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

# Importer nos modules
from utils import vocabulary, preprocessing, data_loader
from models import caption_model

from config import CONFIG


class Trainer:
    """
    Classe pour gérer l'entraînement du modèle
    """
    
    def __init__(self, model, train_loader, val_loader, vocabulary, config):
        """
        Args:
            model (ImageCaptioningModel): Modèle à entraîner
            train_loader (DataLoader): DataLoader d'entraînement
            val_loader (DataLoader): DataLoader de validation
            vocabulary (Vocabulary): Vocabulaire
            config (dict): Configuration de l'entraînement
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        
        # Device (GPU si disponible)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation de: {self.device}")
        
        # Déplacer le modèle sur le device
        self.model.to(self.device)
        
        # Loss function (CrossEntropyLoss)
        # ignore_index pour ignorer le padding dans le calcul de la loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word2idx[vocabulary.pad_token])
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        # Learning rate scheduler (optionnel)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
        
        # Historique de l'entraînement
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Créer les dossiers de sauvegarde
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
    
    def train_epoch(self, epoch):
        """
        Entraîne le modèle pendant une epoch
        
        Args:
            epoch (int): Numéro de l'epoch
            
        Returns:
            float: Loss moyenne de l'epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, (images, captions, lengths) in enumerate(pbar):
            # Déplacer sur le device
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Teacher forcing: input = tous les mots sauf le dernier
            #                  target = tous les mots sauf le premier
            inputs = captions[:, :-1]   # Enlever <END>
            targets = captions[:, 1:]   # Enlever <START>
            
            # Forward pass
            outputs = self.model(images, inputs)  # (batch_size, seq_len-1, vocab_size)
            
            # Reshape pour le calcul de la loss
            # outputs: (batch_size * seq_len, vocab_size)
            # targets: (batch_size * seq_len)
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)
            
            # Calculer la loss
            loss = self.criterion(outputs, targets)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (évite les gradients explosifs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Mise à jour des poids
            self.optimizer.step()
            
            # Accumuler la loss
            total_loss += loss.item()
            
            # Mettre à jour la progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """
        Évalue le modèle sur le set de validation
        
        Returns:
            float: Loss moyenne sur la validation
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader, desc='Validation'):
                # Déplacer sur le device
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # Teacher forcing
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                
                # Forward pass
                outputs = self.model(images, inputs)
                
                # Reshape
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)
                
                # Calculer la loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """
        Boucle d'entraînement complète
        """
        print("\n" + "="*70)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("="*70)
        
        params = self.model.get_num_params()
        print(f"\nNombre de paramètres: {params['total']:,}")
        print(f"  Encoder: {params['encoder']:,}")
        print(f"  Decoder: {params['decoder']:,}")
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Device: {self.device}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Entraînement
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Afficher les résultats
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")

            # Sauvegarder un checkpoint régulier
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'], 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                caption_model.save_model(
                    self.model, 
                    checkpoint_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    vocab=self.vocabulary
                )
            
            # Sauvegarder le meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                caption_model.save_model(
                    self.model, 
                    best_path, 
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    vocab=self.vocabulary
                )
                patience_counter = 0
                print(f"  ✓ Nouveau meilleur modèle sauvegardé (loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ✗ Pas d'amélioration (patience: {patience_counter}/{self.config['patience']})")
                
                # Early stopping
                if patience_counter >= self.config['patience']:
                    print("\nEarly stopping déclenché !")
                    break
            
            
            
        
        # Temps total
        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/60:.2f} minutes")
        print(f"Meilleure validation loss: {self.best_val_loss:.4f}")
        
        # Sauvegarder les courbes d'apprentissage
        self.plot_learning_curves()
        
        # Sauvegarder l'historique
        self.save_history()
    
    def plot_learning_curves(self):
        """
        Trace et sauvegarde les courbes d'apprentissage
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder
        save_path = os.path.join(self.config['log_dir'], 'learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCourbes d'apprentissage sauvegardées dans {save_path}")
        plt.close()
    
    def save_history(self):
        """
        Sauvegarde l'historique d'entraînement en JSON
        """
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Historique sauvegardé dans {save_path}")


def main():
    """
    Fonction principale pour lancer l'entraînement
    """
    

    print("="*70)
    print("PRÉPARATION DES DONNÉES")
    print("="*70)
    
    # ========================================================================
    # CHARGER/CRÉER LE VOCABULAIRE
    # ========================================================================
    
    if os.path.exists(CONFIG['vocab_path']):
        print(f"\nChargement du vocabulaire depuis {CONFIG['vocab_path']}")
        vocab = vocabulary.Vocabulary.load(CONFIG['vocab_path'])
    else:
        print("\nConstruction du vocabulaire...")
        caption_prep = preprocessing.CaptionPreprocessor(
            CONFIG['captions_file'],
            CONFIG['images_dir']
        )
        
        vocab = vocabulary.Vocabulary(freq_threshold=CONFIG['freq_threshold'])
        vocab.build_vocabulary(caption_prep.get_all_captions())
        vocab.save(CONFIG['vocab_path'])
    
    vocab_size = len(vocab)
    print(f"Taille du vocabulaire: {vocab_size}")
    
    # ========================================================================
    # PRÉPARER LES DONNÉES
    # ========================================================================
    
    print("\nPréparation des données...")
    caption_prep = preprocessing.CaptionPreprocessor(
        CONFIG['captions_file'],
        CONFIG['images_dir']
    )
    
    splits = caption_prep.split_data(
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        random_seed=CONFIG['random_seed']
    )
    
    # Préprocesseur d'images
    image_prep = preprocessing.ImagePreprocessor(
        image_size=CONFIG['image_size'],
        normalize=True
    )
    
    # Créer les DataLoaders
    train_loader, val_loader = data_loader.get_data_loaders(
        train_pairs=splits['train'],
        val_pairs=splits['val'],
        vocabulary=vocab,
        image_preprocessor=image_prep,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        shuffle_train=True
    )
    
    # ========================================================================
    # CRÉER LE MODÈLE
    # ========================================================================
    
    print("\nCréation du modèle...")
    model = caption_model.create_model(
        vocab_size=vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        feature_dim=CONFIG['feature_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        encoder_type=CONFIG['encoder_type']
    )
    
    # ========================================================================
    # ENTRAÎNER
    # ========================================================================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocab,
        config=CONFIG
    )
    
    trainer.train()
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ !")
    print("="*70)


if __name__ == "__main__":
    main()