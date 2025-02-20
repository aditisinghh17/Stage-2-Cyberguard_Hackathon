# train_model.py
import os
import numpy as np
import pandas as pd
from crime_classifier import CrimeClassifier
import matplotlib.pyplot as plt

def train():
    print("Starting training process...")
    
    # Initialize the classifier
    classifier = CrimeClassifier(
        model_path='saved_models/bilstm_model',
        tokenizer_path='saved_models/tokenizer.pkl',
        label_encoder_path='saved_models/label_encoder.pkl'
    )
    
    # Create directory for saved models if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    try:
        # Load data
        print("Loading data...")
        df = classifier.load_data('classified_crimes_with_subcategories.csv')
        print(f"Loaded {len(df)} records")
        
        # Load GloVe embeddings
        print("Loading GloVe embeddings...")
        embedding_index = classifier.load_embeddings('glove.6B.100d.txt')
        print("GloVe embeddings loaded")
        
        # Prepare data
        print("Preparing data...")
        X_train, X_test, y_train, y_test = classifier.prepare_data()
        print("Data preparation completed")
        
        # Save test data for later evaluation
        np.save('saved_models/X_test.npy', X_test)
        np.save('saved_models/y_test.npy', y_test)
        
        # Build model
        print("Building model...")
        model = classifier.build_model()
        print("Model architecture:")
        model.summary()
        
        # Train model
        print("\nStarting model training...")
        history = classifier.train(
            X_train, 
            y_train, 
            X_test, 
            y_test,
            epochs=5,  # Adjust as needed
            batch_size=32
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = classifier.evaluate(X_test, y_test)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save model and preprocessing objects
        print("\nSaving model and preprocessing objects...")
        classifier.save_model()
        print("Model saved successfully")
        
        # Plot and save training history
        print("\nPlotting training history...")
        classifier.plot_training_history(history)
        plt.savefig('saved_models/training_history.png')
        plt.close()
        
        print("\nTraining process completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train()
