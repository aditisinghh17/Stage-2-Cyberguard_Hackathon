import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils.np_utils import to_categorical

# Parameters
embedding_dim = 100  # Adjust based on your embedding.txt
max_sequence_length = 1000  # Adjust based on your data
num_classes = 57

# Load your data
# Assume 'texts' is a list of text samples and 'labels' is a list of label ids
# texts = [...]
# labels = [...]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens.')

# Pad sequences
data = pad_sequences(sequences, maxlen=max_sequence_length)

# One-hot encode labels
labels = to_categorical(np.asarray(labels), num_classes=num_classes)

# Load embedding.txt into a dictionary
embeddings_index = {}
with open('embedding.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print(f'Found {len(embeddings_index)} word vectors in embedding.txt.')

# Prepare embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)