import numpy as np
from collections import defaultdict
import re
from typing import List, Dict, Set, Tuple
import random

class FastTextModel:
    def __init__(self, 
                 embedding_dim: int = 100, 
                 min_count: int = 5,
                 min_ngram: int = 3,
                 max_ngram: int = 6,
                 learning_rate: float = 0.05,
                 context_window: int = 5):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.learning_rate = learning_rate
        self.context_window = context_window
        
        # Vocabularies for words and subwords
        self.word_vocab = {}
        self.subword_vocab = {}
        self.reverse_word_vocab = {}
        self.reverse_subword_vocab = {}
        
        # Embeddings for words and subwords
        self.word_embeddings = None
        self.subword_embeddings = None
        
        # Word frequency counter
        self.word_counts = defaultdict(int)

    def generate_ngrams(self, word: str) -> Set[str]:
        """Generate character n-grams for a word"""
        word = f"<{word}>"  # Add boundary markers
        ngrams = set()
        
        for n in range(self.min_ngram, min(len(word), self.max_ngram + 1)):
            for i in range(len(word) - n + 1):
                ngrams.add(word[i:i + n])
                
        return ngrams

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for mixed English-Hindi-Hinglish content"""
        # Convert to lowercase
        text = text.lower()
        
        # Handle special characters while preserving meaningful punctuation
        text = re.sub(r'[^\w\s@#.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().split()

    def build_vocabularies(self, texts: List[str]):
        """Build vocabularies for words and subwords"""
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
                self.reverse_word_vocab[word_idx] = word
                word_idx += 1
        
        # Build subword vocabulary
        subword_idx = 0
        all_ngrams = set()
        
        for word in self.word_vocab:
            ngrams = self.generate_ngrams(word)
            all_ngrams.update(ngrams)
        
        for ngram in sorted(all_ngrams):  # Sort for deterministic vocabulary
            self.subword_vocab[ngram] = subword_idx
            self.reverse_subword_vocab[subword_idx] = ngram
            subword_idx += 1
        
        # Initialize embeddings
        self.word_embeddings = np.random.normal(0, 0.1, 
                                              (len(self.word_vocab), self.embedding_dim))
        self.subword_embeddings = np.random.normal(0, 0.1, 
                                                 (len(self.subword_vocab), self.embedding_dim))

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get the combined word and subword vectors for a word"""
        if word in self.word_vocab:
            word_idx = self.word_vocab[word]
            word_vector = self.word_embeddings[word_idx]
        else:
            word_vector = np.zeros(self.embedding_dim)
        
        # Add subword vectors
        ngrams = self.generate_ngrams(word)
        subword_vectors = []
        
        for ngram in ngrams:
            if ngram in self.subword_vocab:
                subword_idx = self.subword_vocab[ngram]
                subword_vectors.append(self.subword_embeddings[subword_idx])
        
        if subword_vectors:
            subword_vector = np.mean(subword_vectors, axis=0)
            return (word_vector + subword_vector) / 2
        
        return word_vector

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def get_context_pairs(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Generate context pairs for training"""
        pairs = []
        for i, token in enumerate(tokens):
            start = max(0, i - self.context_window)
            end = min(len(tokens), i + self.context_window + 1)
            
            context_words = tokens[start:i] + tokens[i+1:end]
            for context_word in context_words:
                pairs.append((token, context_word))
        
        return pairs

    def train(self, texts: List[str], epochs: int = 5):
        """Train the FastText model"""
        print("Building vocabularies...")
        self.build_vocabularies(texts)
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            pairs_count = 0
            
            for text in texts:
                tokens = self.preprocess_text(text)
                pairs = self.get_context_pairs(tokens)
                
                for target_word, context_word in pairs:
                    # Get target word vector
                    target_vector = self.get_word_vector(target_word)
                    
                    # Positive sample
                    if context_word in self.word_vocab:
                        context_idx = self.word_vocab[context_word]
                        z = np.dot(target_vector, self.word_embeddings[context_idx])
                        p = self.sigmoid(z)
                        loss = -np.log(p)
                        
                        # Update embeddings
                        grad = (p - 1) * target_vector
                        self.word_embeddings[context_idx] -= self.learning_rate * grad
                        
                        # Update subword embeddings
                        for ngram in self.generate_ngrams(target_word):
                            if ngram in self.subword_vocab:
                                subword_idx = self.subword_vocab[ngram]
                                self.subword_embeddings[subword_idx] -= self.learning_rate * grad
                        
                        # Negative sampling
                        for _ in range(5):
                            neg_idx = random.randint(0, len(self.word_vocab) - 1)
                            z_neg = np.dot(target_vector, self.word_embeddings[neg_idx])
                            p_neg = self.sigmoid(z_neg)
                            loss -= np.log(1 - p_neg)
                            
                            grad_neg = p_neg * target_vector
                            self.word_embeddings[neg_idx] -= self.learning_rate * grad_neg
                        
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

    def find_similar_words(self, word: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find n most similar words"""
        word_vector = self.get_word_vector(word)
        similarities = []
        
        for other_word in self.word_vocab:
            if other_word == word:
                continue
            
            other_vector = self.get_word_vector(other_word)
            similarity = np.dot(word_vector, other_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(other_vector))
            similarities.append((other_word, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    def save_model(self, filepath: str):
        """Save model to file"""
        np.savez(filepath,
                word_embeddings=self.word_embeddings,
                subword_embeddings=self.subword_embeddings,
                word_vocab=np.array(list(self.word_vocab.items())),
                subword_vocab=np.array(list(self.subword_vocab.items())),
                params=np.array([self.embedding_dim, self.min_count, 
                               self.min_ngram, self.max_ngram]))
    
    def save_embeddings_text(self, filepath: str, include_subwords: bool = False):
        """
        Save embeddings in text format:
        First line: <vocab_size> <dimension>
        Following lines: word v1 v2 v3 ... vn
        
        Args:
            filepath: Path to save the embeddings
            include_subwords: Whether to include subword embeddings
        """
        # Calculate total vocabulary size
        vocab_size = len(self.word_vocab)
        if include_subwords:
            vocab_size += len(self.subword_vocab)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"{vocab_size} {self.embedding_dim}\n")
            
            # Write word embeddings
            for word, idx in self.word_vocab.items():
                vector = self.word_embeddings[idx]
                vector_str = ' '.join(map(str, vector))
                f.write(f"{word} {vector_str}\n")
            
            # Write subword embeddings if requested
            if include_subwords:
                for subword, idx in self.subword_vocab.items():
                    vector = self.subword_embeddings[idx]
                    vector_str = ' '.join(map(str, vector))
                    f.write(f"__subword__{subword} {vector_str}\n")
    
    def save_text_embeddings(self, filepath: str, texts: List[str]):
        """
        Save text embeddings in text format:
        First line: <number_of_texts> <dimension>
        Following lines: text_id v1 v2 v3 ... vn
        
        Args:
            filepath: Path to save the embeddings
            texts: List of texts to generate embeddings for
        """
        embeddings = [self.get_text_embedding(text) for text in texts]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"{len(texts)} {self.embedding_dim}\n")
            
            # Write embeddings
            for i, embedding in enumerate(embeddings):
                vector_str = ' '.join(map(str, embedding))
                f.write(f"text_{i} {vector_str}\n")

    @classmethod
    def load_model(cls, filepath: str):
        """Load model from file"""
        data = np.load(filepath, allow_pickle=True)
        
        # Create model with saved parameters
        params = data['params']
        model = cls(embedding_dim=int(params[0]),
                   min_count=int(params[1]),
                   min_ngram=int(params[2]),
                   max_ngram=int(params[3]))
        
        # Load vocabularies
        model.word_vocab = dict(data['word_vocab'])
        model.subword_vocab = dict(data['subword_vocab'])
        model.reverse_word_vocab = {v: k for k, v in model.word_vocab.items()}
        model.reverse_subword_vocab = {v: k for k, v in model.subword_vocab.items()}
        
        # Load embeddings
        model.word_embeddings = data['word_embeddings']
        model.subword_embeddings = data['subword_embeddings']
        
        return model

# Example usage:
def demo_fasttext():
    # Sample cybercrime descriptions (mixed English-Hindi-Hinglish)
    sample_texts = [
        "mere account se paise transfer ho gaye without OTP",
        "fake shopping website se order kiya but product nahi mila",
        "someone hacked my email and sent spam to contacts",
        "UPI fraud karke mere bank account se paise nikal liye",
        "received suspicious whatsapp message with malicious link",
        "mobile pe fake banking app download kiya"
    ]
    
    # Initialize and train model
    model = FastTextModel(
        embedding_dim=50,
        min_count=1,
        min_ngram=3,
        max_ngram=6
    )
    
    # Train the model
    model.train(sample_texts, epochs=10)
    
    # Get embedding for a text
    text = "fake bank app fraud kiya"
    embedding = model.get_text_embedding(text)
    
    # Find similar words
    similar_words = model.find_similar_words("fraud")
    
    # Save model
    model.save_model("cybercrime_fasttext.npz")
    
    # Load model
    loaded_model = FastTextModel.load_model("cybercrime_fasttext.npz")
    
    # Save embeddings in text format
    model.save_embeddings_text(
        filepath="cybercrime_word_embeddings.txt",
        include_subwords=True  # Set to True to include subword embeddings
    )
    
    # Save text embeddings for the sample texts
    model.save_text_embeddings(
        filepath="cybercrime_text_embeddings.txt",
        texts=sample_texts
    )
    
    return model, embedding, similar_words