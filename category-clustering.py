import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

class CategoryClusterer:
    def __init__(self, embedding_dim: int = 300, encoding_dim: int = 128,
                 n_clusters: int = 61):
        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim
        self.n_clusters = n_clusters
        self.autoencoder = None
        self.encoder = None
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def build_autoencoder(self) -> None:
        """Build the autoencoder architecture with additional improvements"""
        input_layer = Input(shape=(self.embedding_dim,))
        
        # Encoder
        x = Dense(256, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(self.encoding_dim, activation='relu')(x)
        encoded = BatchNormalization()(x)
        
        # Decoder
        x = Dense(256, activation='relu')(encoded)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        decoded = Dense(self.embedding_dim, activation='sigmoid')(x)
        
        # Create and compile the autoencoder
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                               loss='mean_squared_error')
        
        # Create encoder model
        self.encoder = Model(input_layer, encoded)
        
    def preprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Preprocess the embeddings with standardization"""
        return self.scaler.fit_transform(embeddings)
        
    def train(self, embeddings: np.ndarray, epochs: int = 50,
              batch_size: int = 32) -> None:
        """Train the autoencoder with early stopping"""
        if self.autoencoder is None:
            self.build_autoencoder()
            
        preprocessed_embeddings = self.preprocess_embeddings(embeddings)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.autoencoder.fit(
            preprocessed_embeddings,
            preprocessed_embeddings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            shuffle=True
        )
        
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform clustering on encoded embeddings"""
        preprocessed_embeddings = self.preprocess_embeddings(embeddings)
        encoded_embeddings = self.encoder.predict(preprocessed_embeddings)
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        
        return self.kmeans.fit_predict(encoded_embeddings)
    
    def visualize_clusters(self, embeddings: np.ndarray,
                          labels: np.ndarray,
                          category_mapping: Dict[int, str] = None) -> None:
        """Visualize clusters using PCA"""
        preprocessed_embeddings = self.preprocess_embeddings(embeddings)
        encoded_embeddings = self.encoder.predict(preprocessed_embeddings)
        
        pca = PCA(n_components=2)
        encoded_embeddings_2d = pca.fit_transform(encoded_embeddings)
        
        plt.figure(figsize=(15, 10))
        scatter = plt.scatter(
            encoded_embeddings_2d[:, 0],
            encoded_embeddings_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=50,
            alpha=0.6
        )
        
        plt.title("Category Clusters in Encoded Space")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        
        if category_mapping:
            # Add legend with category names
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=scatter.cmap(scatter.norm(i)),
                                        label=category_mapping[i], markersize=10)
                             for i in range(self.n_clusters)]
            plt.legend(handles=legend_elements, loc='center left',
                      bbox_to_anchor=(1, 0.5))
        
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    def get_cluster_summary(self, texts: List[str], labels: np.ndarray,
                           category_mapping: Dict[int, str] = None) -> pd.DataFrame:
        """Create a summary DataFrame of the clustering results"""
        df = pd.DataFrame({
            'text': texts,
            'cluster': labels
        })
        
        if category_mapping:
            df['category'] = df['cluster'].map(category_mapping)
            
        return df

# Example usage:
def main():
    # Create your category mapping
    category_mapping = {
        # Main categories (0-3)
        0: "Category A",
        1: "Category B",
        2: "Category C",
        3: "Category D",
        # Subcategories (4-60)
        4: "Subcategory A1",
        # ... add all your categories
    }
    
    # Load your embeddings and texts
    # embeddings = load_embeddings()  # Shape: (n_samples, embedding_dim)
    # texts = load_texts()  # List of corresponding texts
    
    # Initialize and train the clusterer
    clusterer = CategoryClusterer(
        embedding_dim=300,  # Adjust based on your embeddings
        encoding_dim=128,
        n_clusters=len(category_mapping)
    )
    
    # Train the model
    clusterer.train(embeddings)
    
    # Get cluster labels
    labels = clusterer.cluster(embeddings)
    
    # Visualize results
    clusterer.visualize_clusters(embeddings, labels, category_mapping)
    
    # Get summary DataFrame
    results_df = clusterer.get_cluster_summary(texts, labels, category_mapping)
    print(results_df.head())
    
    # Analyze cluster distribution
    category_distribution = results_df['category'].value_counts()
    print("\nCategory Distribution:")
    print(category_distribution)

if __name__ == "__main__":
    main()
