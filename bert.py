import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Define the dataset class
class TextDataset(Dataset):
    def _init_(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def _len_(self):
        return len(self.embeddings)

    def _getitem_(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Define the model
class BertClassifier(nn.Module):
    def _init_(self, embedding_dim, num_classes):
        super(BertClassifier, self)._init_()
        self.lstm = nn.LSTM(embedding_dim, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out[:, -1, :])  # Take the output of the last LSTM cell
        return logits

# Load your dataset
# Make sure embeddings are precomputed and saved as a file
embeddings = pd.read_csv('your_embeddings.csv').values  # Replace with your embeddings file
labels = pd.read_csv('your_labels.csv')['sub_category'].values  # Replace with your labels file

# Create datasets and data loaders
from sklearn.model_selection import train_test_split
train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(embeddings, labels, test_size=0.2)
train_dataset = TextDataset(train_embeddings, train_labels)
val_dataset = TextDataset(val_embeddings, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model, loss, and optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
embedding_dim = embeddings.shape[1]
model = BertClassifier(embedding_dim=embedding_dim, num_classes=57).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for embeddings, labels in train_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Save the model
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/bert_classifier.pth')
print("Model saved!")

# To load the model later
# model.load_state_dict(torch.load('model/bert_classifier.pth'))
# model.eval()