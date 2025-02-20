import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class CrimeClassifier:
    def __init__(self, model_path='bilstm_model', 
                 tokenizer_path='tokenizer.pkl',
                 label_encoder_path='label_encoder.pkl'):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.max_words = 5000
        self.max_len = 200
        self.embedding_dim = 100
        
        # Initialize GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU support enabled")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found. Running on CPU")

    def load_data(self, data_path):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(data_path)
        return self.df

    def load_embeddings(self, glove_path):
        """Load GloVe embeddings"""
        self.embedding_index = {}
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                self.embedding_index[word] = coefs
        return self.embedding_index

    def prepare_data(self):
        """Prepare data for training"""
        # Tokenization
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.df['crimeaditionalinfo'])
        sequences = self.tokenizer.texts_to_sequences(self.df['crimeaditionalinfo'])
        self.padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        # Label encoding
        self.label_encoder = LabelEncoder()
        self.df['category_encoded'] = self.label_encoder.fit_transform(self.df['category'])

        # Prepare embedding matrix
        self.embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if i < self.max_words:
                embedding_vector = self.embedding_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector

        # Calculate class weights
        classes = np.unique(self.df['category'])
        class_weights = compute_class_weight('balanced', 
                                          classes=classes, 
                                          y=self.df['category'])
        self.class_weights_dict = dict(zip(classes, class_weights))

        # Train-test split
        return train_test_split(
            self.padded_sequences, 
            self.df['category_encoded'], 
            test_size=0.2, 
            random_state=42
        )

    def build_model(self):
        """Build the BiLSTM model"""
        self.model = Sequential([
            Embedding(self.max_words, 64, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(32, activation='relu'),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return self.model

    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=batch_size
        )
        return history

    def save_model(self):
        """Save the model and preprocessing objects"""
        self.model.save(self.model_path)
        import pickle
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(self.label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def load_saved_model(self):
        """Load the saved model and preprocessing objects"""
        import pickle
        self.model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict_real_time(self, text, semantic_checker):
        """Make real-time predictions with semantic checking"""
        # First apply semantic checking
        cleaned_text = semantic_checker.process_text(text)
        
        # Tokenize and pad the text
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        
        # Make prediction
        prediction = self.model.predict(padded)
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction[0])])
        
        confidence = np.max(prediction[0])
        return predicted_class[0], confidence

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_labels),
            'precision': precision_score(y_test, y_pred_labels, average='weighted'),
            'recall': recall_score(y_test, y_pred_labels, average='weighted'),
            'f1': f1_score(y_test, y_pred_labels, average='weighted')
        }
        return metrics

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Over Epochs")
        plt.legend()
        plt.show()
