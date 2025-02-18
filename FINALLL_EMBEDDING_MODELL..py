import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

# Tokenizer Class
class CharLevelSubwordTokenizer:
    def __init__(self, max_subword_length=3):
        self.max_subword_length = max_subword_length
        self.piece2idx = {'<unk>': 0}  # Ensure <unk> token is in vocabulary
        self.idx2piece = {0: '<unk>'}

    def train(self, corpus):
        subword_freq = {}
        
        for word in corpus:
            for length in range(1, self.max_subword_length + 1):
                for i in range(len(word) - length + 1):
                    subword = word[i:i + length]
                    subword_freq[subword] = subword_freq.get(subword, 0) + 1
        
        sorted_subwords = sorted(subword_freq.items(), key=lambda x: x[1], reverse=True)
        
        for i, (subword, _) in enumerate(sorted_subwords, start=1):
            self.piece2idx[subword] = i
            self.idx2piece[i] = subword

    def tokenize(self, word):
        tokens = []
        i = 0
        
        while i < len(word):
            found = False
            for length in range(self.max_subword_length, 0, -1):
                if i + length <= len(word):
                    subword = word[i:i + length]
                    if subword in self.piece2idx:
                        tokens.append(self.piece2idx[subword])
                        i += length
                        found = True
                        break
            if not found:
                tokens.append(self.piece2idx['<unk>'])  # Assign <unk> if no subword match
                i += 1
        return tokens

# Skip-gram Model with Character-Level CNN Embeddings
class SkipGramCharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, max_subword_length=3):
        super(SkipGramCharCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.char_cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=max_subword_length, padding=1)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        target_embeds = self.target_embeddings(target)  # (batch_size, embedding_dim)
        context_embeds = self.context_embeddings(context)  # (batch_size, embedding_dim)
        context_embeds = self.char_cnn(context_embeds.unsqueeze(1)).squeeze(1)  # Apply CNN
        context_embeds = context_embeds.mean(dim=2)  # Aggregate information
        score = torch.sum(target_embeds * context_embeds, dim=1)
        return score

# Training Function
def train_model(corpus, embedding_dim=100, context_size=2, epochs=10, lr=0.01):
    tokenizer = CharLevelSubwordTokenizer()
    tokenizer.train(corpus)
    
    vocab_size = len(tokenizer.piece2idx)
    model = SkipGramCharCNN(vocab_size, embedding_dim, context_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    data = []
    for word in corpus:
        tokenized_word = tokenizer.tokenize(word)
        for i in range(len(tokenized_word)):
            for j in range(max(0, i - context_size), min(len(tokenized_word), i + context_size + 1)):
                if i != j:
                    data.append((tokenized_word[i], tokenized_word[j], 1))  # Positive sample
                    neg_context = random.choice(list(set(range(vocab_size)) - {tokenized_word[i]}))
                    data.append((tokenized_word[i], neg_context, 0))  # Negative sample
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)
        
        for target, context, label in data:
            target = torch.tensor([target], dtype=torch.long)
            context = torch.tensor([context], dtype=torch.long)
            label = torch.tensor([label], dtype=torch.float)
            
            optimizer.zero_grad()
            score = model(target, context)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')
    
    save_model(model)
    save_tokenizer(tokenizer)
    return model, tokenizer

# Save & Load Functions
def save_model(model, path='skipgram_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(vocab_size, embedding_dim, context_size, path='skipgram_model.pth'):
    model = SkipGramCharCNN(vocab_size, embedding_dim, context_size)
    model.load_state_dict(torch.load(path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

def save_tokenizer(tokenizer, path='tokenizer.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path='tokenizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Generate Word Embeddings
def get_corpus_embeddings(model, tokenizer, corpus, save_path='embeddings.npy'):
    embeddings = {}
    for word in corpus:
        token_ids = tokenizer.tokenize(word)
        token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            word_embedding = model.target_embeddings(token_tensor).mean(dim=1).numpy()
        embeddings[word] = word_embedding
    np.savez_compressed(save_path, **embeddings)

# Example Usage
if __name__ == "__main__":
    sample_corpus = ['apple', 'banana', 'grape', 'orange', 'melon']
    trained_model, trained_tokenizer = train_model(sample_corpus)
    get_corpus_embeddings(trained_model, trained_tokenizer, sample_corpus)
    print("Model Trained and Embeddings Generated!")