import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class MultiStageCybercrimeClassifier:
    def __init__(self, 
                 embedding_dim: int,
                 n_categories: int,
                 n_subcategories: Dict[int, int],
                 encoded_dim: int = 128,
                 sequence_length: int = 10):
        """
        Initialize the multi-stage classifier.
        
        Args:
            embedding_dim: Dimension of input embeddings
            n_categories: Number of main categories
            n_subcategories: Dict mapping category index to number of subcategories
            encoded_dim: Dimension of encoded representation
            sequence_length: Sequence length for BiLSTM
        """
        self.embedding_dim = embedding_dim
        self.n_categories = n_categories
        self.n_subcategories = n_subcategories
        self.encoded_dim = encoded_dim
        self.sequence_length = sequence_length
        
        self.autoencoder = None
        self.encoder = None
        self.bilstm_model = None
        self.scaler = StandardScaler()
        
    def build_autoencoder(self) -> None:
        """
        Build autoencoder for embedding compression and feature learning.
        """
        # Encoder
        input_layer = Input(shape=(self.embedding_dim,))
        
        x = Dense(256, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(self.encoded_dim, activation='relu')(x)
        encoded = BatchNormalization()(x)
        
        # Decoder
        x = Dense(256, activation='relu')(encoded)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        decoded = Dense(self.embedding_dim, activation='sigmoid')(x)
        
        # Create models
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # Compile
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
    
    def build_bilstm_model(self) -> None:
        """
        Build BiLSTM model for sequence learning and refinement.
        """
        input_layer = Input(shape=(self.sequence_length, self.encoded_dim))
        
        x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(LSTM(64))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        
        outputs = Dense(self.n_categories, activation='softmax')(x)
        
        self.bilstm_model = Model(input_layer, outputs)
        self.bilstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare sequences for BiLSTM processing.
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def cluster_data(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Perform hierarchical clustering on encoded embeddings.
        """
        # Main category clustering using HDBSCAN
        main_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.3,
            prediction_data=True
        )
        main_labels = main_clusterer.fit_predict(embeddings)
        
        # Subcategory clustering
        sub_labels = {}
        for cat in range(self.n_categories):
            mask = main_labels == cat
            if np.sum(mask) > 0:
                sub_data = embeddings[mask]
                n_subclusters = self.n_subcategories.get(cat, 2)
                
                sub_clusterer = KMeans(
                    n_clusters=n_subclusters,
                    random_state=42
                )
                sub_labels[cat] = sub_clusterer.fit_predict(sub_data)
        
        return main_labels, sub_labels
    
    def fit(self, embeddings: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train the complete multi-stage model.
        """
        print("Stage 1: Training Autoencoder...")
        if self.autoencoder is None:
            self.build_autoencoder()
        
        # Scale embeddings
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Train autoencoder
        self.autoencoder.fit(
            scaled_embeddings,
            scaled_embeddings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Get encoded representations
        encoded_data = self.encoder.predict(scaled_embeddings)
        
        print("\nStage 2: Performing Initial Clustering...")
        main_labels, sub_labels = self.cluster_data(encoded_data)
        
        print("\nStage 3: Training BiLSTM Model...")
        if self.bilstm_model is None:
            self.build_bilstm_model()
        
        # Prepare sequences for BiLSTM
        sequences = self.prepare_sequences(encoded_data)
        
        # Prepare labels (using main categories)
        sequence_labels = main_labels[self.sequence_length-1:]
        labels_onehot = tf.keras.utils.to_categorical(sequence_labels, self.n_categories)
        
        # Train BiLSTM
        self.bilstm_model.fit(
            sequences,
            labels_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Predict categories and subcategories for new data.
        """
        # Scale and encode embeddings
        scaled_embeddings = self.scaler.transform(embeddings)
        encoded_data = self.encoder.predict(scaled_embeddings)
        
        # Prepare sequences
        sequences = self.prepare_sequences(encoded_data)
        
        # Get refined categories from BiLSTM
        refined_probs = self.bilstm_model.predict(sequences)
        refined_categories = np.argmax(refined_probs, axis=1)
        
        # Extend predictions to match original length
        refined_categories = np.pad(
            refined_categories,
            (self.sequence_length-1, 0),
            mode='edge'
        )
        
        # Get subcategories
        _, sub_labels = self.cluster_data(encoded_data)
        
        return refined_categories, sub_labels
    
    def visualize_results(self, embeddings: np.ndarray,
                         main_labels: np.ndarray,
                         category_names: Dict[int, str] = None):
        """
        Visualize the clustering results.
        """
        # Get encoded representations
        scaled_embeddings = self.scaler.transform(embeddings)
        encoded_data = self.encoder.predict(scaled_embeddings)
        
        # Use t-SNE for visualization
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        vis_data = tsne.fit_transform(encoded_data)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        scatter = plt.scatter(vis_data[:, 0], vis_data[:, 1],
                            c=main_labels, cmap='tab20', alpha=0.6)
        
        if category_names:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=scatter.cmap(scatter.norm(i)),
                                        label=category_names.get(i, f"Category {i}"),
                                        markersize=10)
                             for i in range(self.n_categories)]
            plt.legend(handles=legend_elements, loc='center left',
                      bbox_to_anchor=(1, 0.5))
        
        plt.title("Multi-stage Clustering Results")
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Sample data (replace with your actual embeddings)
    embedding_dim = 300
    n_samples = 1000
    embeddings = np.random.rand(n_samples, embedding_dim)
    
    # Define category structure
    n_categories = 4
    n_subcategories = {
        0: 3,
        1: 4,
        2: 3,
        3: 2
    }
    
    # Initialize classifier
    classifier = MultiStageCybercrimeClassifier(
        embedding_dim=embedding_dim,
        n_categories=n_categories,
        n_subcategories=n_subcategories,
        encoded_dim=128,
        sequence_length=10
    )
    
    # Train the model
    classifier.fit(embeddings, epochs=20)
    
    # Get predictions
    main_labels, sub_labels = classifier.predict(embeddings)
    
    # Visualize results
    category_names = {
        0: "Network Attacks",
        1: "Malware",
        2: "Social Engineering",
        3: "Data Breaches"
    }
    classifier.visualize_results(embeddings, main_labels, category_names)

if __name__ == "__main__":
    main()
