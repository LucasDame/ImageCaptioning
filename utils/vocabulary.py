"""
Vocabulary Builder for Image Captioning
Gère la construction du vocabulaire à partir des captions et la conversion mots <-> indices
"""

import pickle
from collections import Counter
import os


class Vocabulary:
    """
    Classe pour construire et gérer le vocabulaire des captions
    """
    
    def __init__(self, freq_threshold=5):
        """
        Args:
            freq_threshold (int): Fréquence minimale pour qu'un mot soit inclus dans le vocabulaire
                                 Les mots moins fréquents seront remplacés par <UNK>
        """
        self.freq_threshold = freq_threshold
        
        # Tokens spéciaux (indices fixes)
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Dictionnaires de mapping
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Initialiser avec les tokens spéciaux
        self._init_vocab()
    
    def _init_vocab(self):
        """
        Initialise le vocabulaire avec les tokens spéciaux
        Important : L'ordre est crucial car il définit les indices
        """
        special_tokens = [
            self.pad_token,    # idx = 0 (utilisé pour le padding)
            self.start_token,  # idx = 1 (début de séquence)
            self.end_token,    # idx = 2 (fin de séquence)
            self.unk_token     # idx = 3 (mots inconnus)
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def build_vocabulary(self, captions_list):
        """
        Construit le vocabulaire à partir d'une liste de captions
        
        Args:
            captions_list (list): Liste de strings, chaque string est une caption
                                 Exemple: ["a dog running", "two cats sitting"]
        """
        print("Construction du vocabulaire...")
        
        # Compter la fréquence de chaque mot
        for caption in captions_list:
            tokens = self._tokenize(caption)
            self.word_freq.update(tokens)
        
        # Ajouter les mots qui dépassent le seuil de fréquence
        idx = len(self.word2idx)  # Commence après les tokens spéciaux
        
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
        
        print(f"Vocabulaire construit avec {len(self.word2idx)} mots")
        print(f"Tokens spéciaux : {self.pad_token}(0), {self.start_token}(1), {self.end_token}(2), {self.unk_token}(3)")
        print(f"Seuil de fréquence : {self.freq_threshold}")
        print(f"Mots filtrés : {len([w for w, f in self.word_freq.items() if f < self.freq_threshold])}")
    
    def _tokenize(self, text):
        """
        Tokenize un texte en mots
        
        Args:
            text (str): Texte à tokenizer
            
        Returns:
            list: Liste de mots en minuscules
        """
        # Convertir en minuscules et enlever la ponctuation
        text = text.lower()
        # Remplacer la ponctuation par des espaces
        for char in ['.', ',', '!', '?', ';', ':', '"', "'", '-', '(', ')']:
            text = text.replace(char, ' ')
        
        # Séparer en mots
        tokens = text.split()
        return tokens
    
    def numericalize(self, text):
        """
        Convertit une caption (texte) en séquence d'indices
        
        Args:
            text (str): Caption à convertir
            
        Returns:
            list: Liste d'indices correspondant aux mots
                  Exemple: "a dog" -> [1, 45, 123, 2] (START, a, dog, END)
        """
        tokens = self._tokenize(text)
        
        # Ajouter START au début et END à la fin
        numericalized = [self.word2idx[self.start_token]]
        
        for token in tokens:
            # Si le mot est dans le vocabulaire, utiliser son index
            # Sinon, utiliser l'index de <UNK>
            if token in self.word2idx:
                numericalized.append(self.word2idx[token])
            else:
                numericalized.append(self.word2idx[self.unk_token])
        
        numericalized.append(self.word2idx[self.end_token])
        
        return numericalized
    
    def denumericalize(self, indices):
        """
        Convertit une séquence d'indices en texte
        
        Args:
            indices (list ou tensor): Liste d'indices
            
        Returns:
            str: Caption reconstruite
        """
        # Si c'est un tensor PyTorch, convertir en liste
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        words = []
        for idx in indices:
            # Arrêter au token END
            if idx == self.word2idx[self.end_token]:
                break
            # Ne pas inclure START et PAD
            if idx not in [self.word2idx[self.start_token], 
                          self.word2idx[self.pad_token]]:
                words.append(self.idx2word[idx])
        
        return ' '.join(words)
    
    def __len__(self):
        """
        Retourne la taille du vocabulaire
        Utilisé pour définir la taille de l'embedding layer
        """
        return len(self.word2idx)
    
    def save(self, filepath):
        """
        Sauvegarde le vocabulaire sur disque
        
        Args:
            filepath (str): Chemin où sauvegarder le vocabulaire
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Vocabulaire sauvegardé dans {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Charge un vocabulaire depuis le disque
        
        Args:
            filepath (str): Chemin du fichier vocabulaire
            
        Returns:
            Vocabulary: Instance du vocabulaire chargé
        """
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulaire chargé depuis {filepath}")
        print(f"Taille du vocabulaire : {len(vocab)}")
        return vocab


# Fonction utilitaire pour tester
if __name__ == "__main__":
    # Exemple d'utilisation
    
    captions = [
        "a dog is running in the park",
        "two cats are sitting on a wall",
        "a dog playing with a ball",
        "the cat is sleeping"
    ]
    
    # Créer et construire le vocabulaire
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(captions)
    
    # Tester la conversion texte -> indices
    test_caption = "a dog is playing"
    indices = vocab.numericalize(test_caption)
    print(f"\nCaption: {test_caption}")
    print(f"Indices: {indices}")
    
    # Tester la conversion indices -> texte
    reconstructed = vocab.denumericalize(indices)
    print(f"Reconstructed: {reconstructed}")
    
    # Sauvegarder
    vocab.save("data/vocab.pkl")