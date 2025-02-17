import numpy as np
import math
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        return torch.matmul(attention, V)
    
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerHybridEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 1024,
                 max_seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
            
        return x

class HinglishDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)

class HybridTransformerModel:
    def __init__(self,
                 base_model,  # Your existing HybridEmbeddingModel
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        self.base_model = base_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create transformer model
        self.transformer = TransformerHybridEmbedding(
            vocab_size=len(base_model.word_vocab),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.transformer.parameters())
        
    def prepare_data(self, texts: List[str]) -> DataLoader:
        dataset = HinglishDataset(texts, self.base_model)
        return DataLoader(dataset, batch_size=32, shuffle=True)
    
    def train(self, texts: List[str], epochs: int = 5):
        dataloader = self.prepare_data(texts)
        self.transformer.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Get base embeddings
                base_embeddings = torch.stack([
                    torch.tensor(self.base_model.get_word_vector(word))
                    for word in batch
                ]).to(self.device)
                
                # Forward pass through transformer
                self.optimizer.zero_grad()
                output = self.transformer(batch)
                
                # Compute loss (using cosine similarity as objective)
                loss = 1 - F.cosine_similarity(output, base_embeddings).mean()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get combined embedding using both base model and transformer"""
        # Get base embedding
        base_embedding = self.base_model.get_text_embedding(text)
        
        # Get transformer embedding
        self.transformer.eval()
        with torch.no_grad():
            tokens = torch.tensor(
                self.base_model.preprocess_text(text)
            ).unsqueeze(0).to(self.device)
            transformer_embedding = self.transformer(tokens).mean(dim=1).squeeze().cpu().numpy()
        
        # Combine embeddings (you can adjust the weights)
        return 0.5 * base_embedding + 0.5 * transformer_embedding
    
    def save_model(self, filepath: str):
        """Save both base model and transformer"""
        # Save base model
        self.base_model.save_embeddings(f"{filepath}_base.txt")
        
        # Save transformer
        torch.save(self.transformer.state_dict(), f"{filepath}_transformer.pt")
        
    @classmethod
    def load_model(cls, filepath: str, base_model):
        """Load a saved model"""
        model = cls(base_model)
        model.transformer.load_state_dict(
            torch.load(f"{filepath}_transformer.pt", map_location=model.device)
        )
        return model

# Example usage
def demo_transformer_hybrid():
    # Sample texts with variations
    texts = [
        "mere account se paise transfer ho gaye without OTP",
        "mere akount se pese transfer hogaya widout otp",
        "mera bank account hack ho gaya",
        "banking fraud ke through paise chori ho gaye",
        "bank ke sath fraud hua hai mere saath",
        "online shopping website pe scam ho gaya",
        "phishing attack me password leak ho gaya",
        "UPI fraud karke mere bank account se paise nikal liye",
        "fake shopping website se order kiya but product nahi mila",
        "someone hacked my email and sent spam to contacts"
    ]
    
    # First initialize and train base model
    from your_previous_code import HybridEmbeddingModel  # Your previous hybrid model
    base_model = HybridEmbeddingModel(embedding_dim=256)
    base_model.train(texts)
    
    # Initialize and train transformer hybrid model
    model = HybridTransformerModel(base_model)
    model.train(texts)
    
    # Test the model
    test_text = "mere bank account me fraud hua hai"
    embedding = model.get_embedding(test_text)
    
    # Save the model
    model.save_model("transformer_hybrid_model")
    
    return model, embedding

def find_similar_texts(model, query_text: str, texts: List[str], n: int = 5):
    """Find n most similar texts to the query text"""
    query_embedding = model.get_embedding(query_text)
    similarities = []
    
    for text in texts:
        text_embedding = model.get_embedding(text)
        similarity = np.dot(query_embedding, text_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
        )
        similarities.append((text, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]