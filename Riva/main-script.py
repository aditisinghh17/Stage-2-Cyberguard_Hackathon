# main.py
from crime_classifier import CrimeClassifier
from semantic_checker import SemanticChecker  # Your previous semantic checker

def train_model():
    # Initialize classifier
    classifier = CrimeClassifier()
    
    # Load and prepare data
    classifier.load_data('classified_crimes_with_subcategories.csv')
    classifier.load_embeddings('glove.6B.100d.txt')
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    
    # Build and train model
    classifier.build_model()
    history = classifier.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    classifier.save_model()
    
    # Plot training history
    classifier.plot_training_history(history)

def make_prediction(text):
    # Initialize classifiers
    classifier = CrimeClassifier()
    semantic_checker = SemanticChecker()  # Your semantic checker
    
    # Load saved model
    classifier.load_saved_model()
    
    # Make prediction
    predicted_class, confidence = classifier.predict_real_time(text, semantic_checker)
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    # For training
    # train_model()
    
    # For prediction
    sample_text = "Your crime description here"
    make_prediction(sample_text)
