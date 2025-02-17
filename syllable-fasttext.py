import re
from typing import List, Set
import numpy as np

class SyllableFastText:
    def __init__(self, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.syllable_vocab = {}
        self.embeddings = None
        self.syllable_embeddings = None

    def get_syllables(self, word: str) -> List[str]:
        """Split word into syllables based on Hindi phonological rules"""
        # Basic Hindi syllable rules for Roman script
        # This is a simplified version and can be made more sophisticated
        word = word.lower()
        
        # Mark vowel clusters
        vowels = 'aeiou'
        word = re.sub(f'[{vowels}]+', lambda m: f'V{len(m.group())}', word)
        
        # Mark consonant clusters
        consonants = 'bcdfghjklmnpqrstvwxyz'
        word = re.sub(f'[{consonants}]+', lambda m: f'C{len(m.group())}', word)
        
        # Split into syllables using common patterns
        syllables = []
        patterns = [
            r'C1V1',  # CV  (like 'ka')
            r'C2V1',  # CCV (like 'kra')
            r'C1V2',  # CVV (like 'kaa')
            r'C1$',   # Final C
            r'V1',    # Just V (like 'a')
            r'V2'     # Long V (like 'aa')
        ]
        
        current_pos = 0
        while current_pos < len(word):
            matched = False
            for pattern in patterns:
                if re.match(pattern, word[current_pos:]):
                    match_len = len(pattern)
                    syllables.append(word[current_pos:current_pos + match_len])
                    current_pos += match_len
                    matched = True
                    break
            if not matched:
                current_pos += 1
        
        return syllables

    def build_vocab(self, texts: List[str]):
        """Build word and syllable vocabularies"""
        word_freqs = {}
        syllable_freqs = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
                
                # Get syllables for each word
                syllables = self.get_syllables(word)
                for syllable in syllables:
                    syllable_freqs[syllable] = syllable_freqs.get(syllable, 0) + 1
        
        # Build vocabularies with indices
        self.vocab = {word: idx for idx, word in enumerate(word_freqs.keys())}
        self.syllable_vocab = {syl: idx for idx, syl in enumerate(syllable_freqs.keys())}
        
        # Initialize embeddings
        self.embeddings = np.random.normal(0, 0.1, 
                                         (len(self.vocab), self.embedding_dim))
        self.syllable_embeddings = np.random.normal(0, 0.1, 
                                                   (len(self.syllable_vocab), self.embedding_dim))

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get combined word and syllable vector"""
        word = word.lower()
        word_vector = np.zeros(self.embedding_dim)
        
        # Get word embedding if it exists
        if word in self.vocab:
            word_vector = self.embeddings[self.vocab[word]]
        
        # Get syllable embeddings
        syllables = self.get_syllables(word)
        syllable_vectors = []
        
        for syllable in syllables:
            if syllable in self.syllable_vocab:
                syllable_vectors.append(
                    self.syllable_embeddings[self.syllable_vocab[syllable]]
                )
        
        if syllable_vectors:
            syllable_vector = np.mean(syllable_vectors, axis=0)
            # Combine word and syllable vectors
            return (word_vector + syllable_vector) / 2
            
        return word_vector

    def train(self, texts: List[str], epochs: int = 5, learning_rate: float = 0.05):
        """Train the model using negative sampling"""
        # Implementation of training logic similar to standard FastText
        # but using syllable-based subwords instead of character n-grams
        # [Training code would go here]
        pass

# Example usage:
def demo_syllable_model():
    texts = [
        "zindagi me bohot mushkil hai",
        "jindagi mein bahut mushkil hai",
        "life me bahot difficult hai"
    ]
    
    model = SyllableFastText(embedding_dim=50)
    model.build_vocab(texts)
    
    # Get embeddings for similar words with different spellings
    v1 = model.get_word_vector("zindagi")
    v2 = model.get_word_vector("jindagi")
    
    # Should be very similar due to syllable-based approach
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"Similarity between zindagi and jindagi: {similarity}")
