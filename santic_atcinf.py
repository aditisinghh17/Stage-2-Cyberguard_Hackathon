import json
import os
import numpy as np
import torch
import torch.nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

# Custom dataset class
class TextMatchingDataset(Dataset):
    def _init_(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def _len_(self):
        return len(self.embeddings)

    def _getitem_(self, item):
        embedding = self.embeddings[item]
        label = self.labels[item]

        return {
            'embedding': torch.tensor(embedding, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Simple semantic matching model
class SemanticMatchingModel(nn.Module):
    def _init_(self, input_size, num_labels):
        super(SemanticMatchingModel, self)._init_()
        self.fc = nn.Linear(input_size, num_labels)

    def forward(self, embedding):
        logits = self.fc(embedding)
        return logits


# Load embeddings and labels
def load_embeddings_and_labels(embedding_path, data_path):
    embeddings = []
    labels = []

    # Load embeddings
    with open(embedding_path, 'r') as f:
        for line in f:
            embeddings.append([float(x) for x in line.strip().split()])

    # Load labels
    with open(data_path, 'r') as file:
        data = json.load(file)
        for item in data:
            labels.append(item['category'])

    return embeddings, labels


# Main training function
def train_model(model, train_loader, val_loader, device, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    return model


# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)

            logits = model(embeddings)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

def save_to_csv(data, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['embedding', 'label'])
        for embedding, label in data:
            writer.writerow([embedding, label])


if _name_ == '_main_':
    # Configurations
    input_size = 768  # Set to the size of your embedding vectors
    batch_size = 16
    epochs = 5
    learning_rate = 2e-5
    num_labels = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your embeddings and labels
    embedding_path = 'embedding.txt'  # Path to your embedding file
    data_path = 'your_data.json'  # Path to your JSON file
    embeddings, labels = load_embeddings_and_labels(embedding_path, data_path)

    # Split into train and validation sets
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Datasets and DataLoaders
    train_dataset = TextMatchingDataset(train_embeddings, train_labels)
    val_dataset = TextMatchingDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = SemanticMatchingModel(input_size, num_labels)

    # Train and evaluate
    model = train_model(model, train_loader, val_loader, device, epochs, learning_rate)
    evaluate_model(model, val_loader, device)

    # Save model
    torch.save(model.state_dict(), 'semantic_matching_model.pth')
    print('Model saved!')
