import numpy as np
from collections import defaultdict
import re
from typing import List, Dict, Set, Tuple
import random

class HybridEmbeddingModel:
    def __init__(self, 
                 embedding_dim: int = 100,
                 min_count: int = 5,
                 context_window: int = 5,
                 learning_rate: float = 0.05):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.context_window = context_window
        self.learning_rate = learning_rate
        
        # Different vocabulary types
        self.word_vocab = {}
        self.syllable_vocab = {}
        self.char_ngram_vocab = {}
        self.phoneme_vocab = {}
        
        # Embeddings for different components
        self.word_embeddings = None
        self.syllable_embeddings = None
        self.char_ngram_embeddings = None
        self.phoneme_embeddings = None
        
        # Frequency counters
        self.word_counts = defaultdict(int)
        
        # Romanization normalization mappings
        self.rom_norm_map = {
            'ph': 'f',
            'gh': 'g',
            'th': 't',
            'dh': 'd',
            'sh': 's',
            'ch': 'c',
            'kh': 'k',
            'z': 'j',
            'w': 'v'
        }

    def normalize_roman(self, text: str) -> str:
        """Normalize different romanization variations"""
        normalized = text.lower()
        for original, replacement in self.rom_norm_map.items():
            normalized = normalized.replace(original, replacement)
        return normalized

    def get_syllables(self, word: str) -> List[str]:
        """Extract syllables using Hindi phonological rules"""
        word = self.normalize_roman(word)
        syllables = []
        
        # Define vowels and consonants
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvxy'
        
        # Basic syllable patterns for Hindi
        i = 0
        while i < len(word):
            syllable = ''
            
            # Handle consonant clusters
            if i < len(word) and word[i] in consonants:
                syllable += word[i]
                i += 1
                if i < len(word) and word[i] in consonants:
                    syllable += word[i]
                    i += 1
            
            # Handle vowels
            if i < len(word) and word[i] in vowels:
                syllable += word[i]
                i += 1
                if i < len(word) and word[i] in vowels:
                    syllable += word[i]
                    i += 1
            
            if syllable:
                syllables.append(syllable)
            else:
                i += 1
                
        return syllables

    def get_char_ngrams(self, word: str, min_n: int = 3, max_n: int = 6) -> Set[str]:
        """Generate character n-grams"""
        word = self.normalize_roman(f"<{word}>")
        ngrams = set()
        
        for n in range(min_n, min(len(word), max_n + 1)):
            for i in range(len(word) - n + 1):
                ngrams.add(word[i:i + n])
                
        return ngrams

    def get_phonemes(self, word: str) -> List[str]:
        """Convert word to approximate phonemes"""
        word = self.normalize_roman(word)
        phonemes = []
        
        # Define phoneme mappings
        vowel_phonemes = {
            'aa': 'ā', 'ee': 'ī', 'oo': 'ū',
            'ai': 'ai', 'au': 'au',
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u'
        }
        
        i = 0
        while i < len(word):
            # Check for two-character vowels first
            if i < len(word) - 1:
                two_chars = word[i:i+2]
                if two_chars in vowel_phonemes:
                    phonemes.append(vowel_phonemes[two_chars])
                    i += 2
                    continue
            
            # Single characters
            if word[i] in vowel_phonemes:
                phonemes.append(vowel_phonemes[word[i]])
            else:
                phonemes.append(word[i])
            i += 1
            
        return phonemes

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for mixed English-Hindi-Hinglish content"""
        # Normalize romanization
        text = self.normalize_roman(text)
        
        # Handle special characters while preserving meaningful punctuation
        text = re.sub(r'[^\w\s@#.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().split()

    def build_vocabularies(self, texts: List[str]):
        """Build vocabularies for all components"""
        # Count word frequencies
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                self.word_counts[token] += 1
        
        # Build word vocabulary
        word_idx = 0
        for word, count in self.word_counts.items():
            if count >= self.min_count:
                self.word_vocab[word] = word_idx
                word_idx += 1
        
        # Build syllable vocabulary
        syllable_set = set()
        for word in self.word_vocab:
            syllables = self.get_syllables(word)
            syllable_set.update(syllables)
        
        self.syllable_vocab = {syl: idx for idx, syl in enumerate(sorted(syllable_set))}
        
        # Build character n-gram vocabulary
        ngram_set = set()
        for word in self.word_vocab:
            ngrams = self.get_char_ngrams(word)
            ngram_set.update(ngrams)
        
        self.char_ngram_vocab = {ng: idx for idx, ng in enumerate(sorted(ngram_set))}
        
        # Build phoneme vocabulary
        phoneme_set = set()
        for word in self.word_vocab:
            phonemes = self.get_phonemes(word)
            phoneme_set.update(phonemes)
        
        self.phoneme_vocab = {ph: idx for idx, ph in enumerate(sorted(phoneme_set))}
        
        # Initialize embeddings
        self.word_embeddings = np.random.normal(0, 0.1, 
                                              (len(self.word_vocab), self.embedding_dim))
        self.syllable_embeddings = np.random.normal(0, 0.1, 
                                                  (len(self.syllable_vocab), self.embedding_dim))
        self.char_ngram_embeddings = np.random.normal(0, 0.1, 
                                                    (len(self.char_ngram_vocab), self.embedding_dim))
        self.phoneme_embeddings = np.random.normal(0, 0.1, 
                                                 (len(self.phoneme_vocab), self.embedding_dim))

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get combined vector using all components"""
        word = self.normalize_roman(word)
        vectors = []
        weights = [0.4, 0.2, 0.2, 0.2]  # Weights for different components
        
        # Word embedding
        if word in self.word_vocab:
            word_idx = self.word_vocab[word]
            vectors.append(self.word_embeddings[word_idx])
        else:
            vectors.append(np.zeros(self.embedding_dim))
        
        # Syllable embeddings
        syllable_vectors = []
        for syllable in self.get_syllables(word):
            if syllable in self.syllable_vocab:
                syllable_idx = self.syllable_vocab[syllable]
                syllable_vectors.append(self.syllable_embeddings[syllable_idx])
        if syllable_vectors:
            vectors.append(np.mean(syllable_vectors, axis=0))
        else:
            vectors.append(np.zeros(self.embedding_dim))
        
        # Character n-gram embeddings
        ngram_vectors = []
        for ngram in self.get_char_ngrams(word):
            if ngram in self.char_ngram_vocab:
                ngram_idx = self.char_ngram_vocab[ngram]
                ngram_vectors.append(self.char_ngram_embeddings[ngram_idx])
        if ngram_vectors:
            vectors.append(np.mean(ngram_vectors, axis=0))
        else:
            vectors.append(np.zeros(self.embedding_dim))
        
        # Phoneme embeddings
        phoneme_vectors = []
        for phoneme in self.get_phonemes(word):
            if phoneme in self.phoneme_vocab:
                phoneme_idx = self.phoneme_vocab[phoneme]
                phoneme_vectors.append(self.phoneme_embeddings[phoneme_idx])
        if phoneme_vectors:
            vectors.append(np.mean(phoneme_vectors, axis=0))
        else:
            vectors.append(np.zeros(self.embedding_dim))
        
        # Combine all vectors with weights
        return np.sum([v * w for v, w in zip(vectors, weights)], axis=0)

    def train_step(self, target_word: str, context_word: str) -> float:
        """Perform a single training step using negative sampling"""
        if target_word not in self.word_vocab or context_word not in self.word_vocab:
            return 0.0
            
        target_idx = self.word_vocab[target_word]
        context_idx = self.word_vocab[context_word]
        
        # Get combined vector for target word
        target_vector = self.get_word_vector(target_word)
        
        # Positive sample
        z = np.dot(target_vector, self.word_embeddings[context_idx])
        p = 1 / (1 + np.exp(-z))
        loss = -np.log(p)
        
        # Update embeddings
        grad = (p - 1) * target_vector
        self.word_embeddings[context_idx] -= self.learning_rate * grad
        
        # Negative sampling
        for _ in range(5):  # Number of negative samples
            neg_idx = random.randint(0, len(self.word_vocab) - 1)
            z_neg = np.dot(target_vector, self.word_embeddings[neg_idx])
            p_neg = 1 / (1 + np.exp(-z_neg))
            loss -= np.log(1 - p_neg)
            
            grad_neg = p_neg * target_vector
            self.word_embeddings[neg_idx] -= self.learning_rate * grad_neg
        
        return loss

    def train(self, texts: List[str], epochs: int = 5):
        """Train the hybrid embedding model"""
        print("Building vocabularies...")
        self.build_vocabularies(texts)
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            pairs_count = 0
            
            for text in texts:
                tokens = self.preprocess_text(text)
                
                for i, token in enumerate(tokens):
                    # Generate context pairs
                    start = max(0, i - self.context_window)
                    end = min(len(tokens), i + self.context_window + 1)
                    
                    context_words = tokens[start:i] + tokens[i+1:end]
                    for context_word in context_words:
                        loss = self.train_step(token, context_word)
                        total_loss += loss
                        pairs_count += 1
            
            avg_loss = total_loss / max(1, pairs_count)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a complete text"""
        tokens = self.preprocess_text(text)
        token_embeddings = []
        
        for token in tokens:
            token_embeddings.append(self.get_word_vector(token))
        
        if not token_embeddings:
            return np.zeros(self.embedding_dim)
        
        return np.mean(token_embeddings, axis=0)

    def save_embeddings(self, filepath: str):
        """Save embeddings in text format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"{len(self.word_vocab)} {self.embedding_dim}\n")
            
            # Write word embeddings
            for word in self.word_vocab:
                vector = self.get_word_vector(word)
                vector_str = ' '.join(map(str, vector))
                f.write(f"{word} {vector_str}\n")

# Example usage
def demo_hybrid_embeddings():
    # Sample texts with variations in spelling
    texts = [
        "mere account se paise transfer ho gaye without OTP",
        "mere akount se pese transfer hogaya widout otp",
        "mera bank account hack ho gaya",
        "banking fraud ke through paise chori ho gaye",
        "bank ke sath fraud hua hai mere saath",
        "online shopping website pe scam ho gaya",
        "phishing attack me password leak ho gaya"
    ]
    
    # Initialize and train model
    model = HybridEmbeddingModel(
        embedding_dim=100,
        min_count=1,
        context_window=5
    )
    
    # Train the model
    model.train(texts, epochs=10)
    
    # Test similar words with different spellings
    word1 = "account"
    word2 = "akount"
    vec1 = model.get_word_vector(word1)
    vec2 = model.get_word_vector(word2)
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
    
    # Save embeddings
    model.save_embeddings("hybrid_embeddings.txt")
    
    return model