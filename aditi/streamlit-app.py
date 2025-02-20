# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from crime_classifier import CrimeClassifier
from semantic_checker import SemanticChecker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class StreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="Crime Classification System",
            page_icon="🔍",
            layout="wide"
        )
        
        # Initialize classifiers
        self.classifier = CrimeClassifier()
        self.semantic_checker = SemanticChecker()
        
        # Load saved model
        try:
            self.classifier.load_saved_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    def run(self):
        st.title("🔍 Crime Classification System")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Choose a page",
            ["Real-time Classification", "Model Performance", "Batch Processing"]
        )
        
        if page == "Real-time Classification":
            self.show_realtime_classification()
        elif page == "Model Performance":
            self.show_model_performance()
        else:
            self.show_batch_processing()

    def show_realtime_classification(self):
        st.header("Real-time Crime Classification")
        
        # Text input
        text_input = st.text_area(
            "Enter crime description:",
            height=150,
            placeholder="Type or paste crime description here..."
        )
        
        if st.button("Classify"):
            if text_input.strip():
                with st.spinner("Processing..."):
                    try:
                        # Make prediction
                        predicted_class, confidence = self.classifier.predict_real_time(
                            text_input, 
                            self.semantic_checker
                        )
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info("Predicted Category")
                            st.subheader(predicted_class)
                            
                        with col2:
                            st.info("Confidence Score")
                            st.progress(confidence)
                            st.text(f"{confidence:.2%}")
                        
                        # Show processed text
                        st.subheader("Processed Text")
                        processed_text = self.semantic_checker.process_text(text_input)
                        st.text_area("", processed_text, height=100)
                        
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")
            else:
                st.warning("Please enter some text to classify.")

    def show_model_performance(self):
        st.header("Model Performance Metrics")
        
        # Load test data
        if st.button("Calculate Performance Metrics"):
            with st.spinner("Calculating metrics..."):
                try:
                    # Load and prepare test data
                    X_test = np.load('X_test.npy')  # You'll need to save these during training
                    y_test = np.load('y_test.npy')
                    
                    # Get predictions
                    y_pred = self.classifier.model.predict(X_test)
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    
                    # Calculate metrics
                    metrics = self.classifier.evaluate(X_test, y_test)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.2%}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1']:.2%}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred_labels)
                    
                    # Plot confusion matrix using seaborn
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title("Confusion Matrix")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(plt)
                    
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")

    def show_batch_processing(self):
        st.header("Batch Processing")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch processing",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                    return
                
                if st.button("Process Batch"):
                    with st.spinner("Processing batch..."):
                        # Process each text
                        results = []
                        for text in df['text']:
                            pred_class, conf = self.classifier.predict_real_time(
                                text,
                                self.semantic_checker
                            )
                            results.append({
                                'text': text,
                                'predicted_class': pred_class,
                                'confidence': conf
                            })
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Batch Processing Results")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "batch_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
