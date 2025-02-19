import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# --- Step 1: Prepare the Entire Corpus for Doc2Vec Training ---
all_documents = df_train_cleaned['crimeaditionalinfo'].astype(str).tolist()

# Create tagged documents for the entire corpus
tagged_all_documents = [TaggedDocument(words=doc.split(), tags=[str(i)]) 
                          for i, doc in enumerate(all_documents)]

# --- Step 2: Train the Doc2Vec Model on the Entire Corpus ---
model = Doc2Vec(vector_size=100,    # Dimensionality of document vectors
                alpha=0.025,        # Initial learning rate
                min_alpha=0.00025,  # Minimum learning rate
                min_count=1,        # Ignores words with total frequency below this
                dm=1)               # dm=1 for Distributed Memory

# Build vocabulary from the entire corpus
model.build_vocab(tagged_all_documents)

# Train for 20 epochs
for epoch in range(20):
    print(f"Epoch {epoch+1}")
    model.train(tagged_all_documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.001
    model.min_alpha = model.alpha

# Optionally save the trained model
model.save("full_corpus_doc2vec.model")
print("Doc2Vec model trained on the entire corpus and saved!")

# --- Step 3: Filter the Data for the Specific Category ---
filtered_df = df_train_cleaned[df_train_cleaned['category'] == 'Online and Social Media Related Crime']
print("Number of documents in filtered category:", filtered_df.shape[0])

# Retrieve the indices
filtered_indices = filtered_df.index.tolist()

# --- Step 4: Extract Document Embeddings for the Filtered Documents ---
filtered_doc_vectors = np.array([model.dv[str(i)] for i in filtered_indices if str(i) in model.dv])
print("Shape of filtered document vectors:", filtered_doc_vectors.shape)

# Now you have document vectors (embeddings) in filtered_doc_vectors
# Each row represents a document's 100-dimensional embedding