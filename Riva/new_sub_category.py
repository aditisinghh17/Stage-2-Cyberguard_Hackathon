import json
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

# Initialize stopwords
stop_words = set(stopwords.words('english'))
indic_stopwords = set([
    "aur", "ki", "hai", "huyi", "ho", "mein", "ye", "ke", "jo", "saath", "ko",
    "bhi", "tatha", "par", "se", "kisi", "un", "apna", "tum", "main", "aap", "inhe",
    "in", "abhi", "ab", "woh", "hum", "unka", "is", "us", "kintu", "athva", "nahin",
    "kar", "firto", "fir", "kese", "esse", "ka", "kabhi", "karna"
])
stop_words = stop_words.union(indic_stopwords)

# Global variables for embeddings
embedding_index = {}
tokenizer = None
embedding_matrix = None
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100

def initialize_embeddings(glove_path):
    """Initialize GloVe embeddings and tokenizer"""
    global embedding_index, tokenizer, embedding_matrix
    
    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = coefs
    
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")

def remove_stopwords(text):
    """Remove stopwords from the given text."""
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def prepare_embedding_matrix(texts):
    """Prepare embedding matrix for the given texts"""
    global embedding_matrix, tokenizer
    
    # Fit tokenizer on texts
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    
    # Create embedding matrix
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < VOCAB_SIZE:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

def text_to_vector(text):
    """Convert text to averaged embedding vector"""
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Get embeddings for each word and average them
    text_embedding = np.zeros(EMBEDDING_DIM)
    count = 0
    for word_idx in padded[0]:
        if word_idx != 0:  # Skip padding
            text_embedding += embedding_matrix[word_idx]
            count += 1
    
    return text_embedding / max(count, 1)  # Avoid division by zero

def load_json(json_file):
    """Load data from a JSON file."""
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_text_representation(entity):
    """Create a processed text representation for categories or subcategories"""
    description = remove_stopwords(entity.get('description', ''))
    transliterated_description = remove_stopwords(entity.get('transliterated_description', ''))
    
    examples = entity.get('related_examples', [])
    filtered_examples = [remove_stopwords(example) for example in examples]
    related_examples = " | ".join(filtered_examples) + " " + " ".join(filtered_examples)
    
    return f"{description} {transliterated_description} {related_examples}".strip()

def classify_description(description, entities):
    """Find the top matching entity (category or subcategory)"""
    filtered_description = remove_stopwords(description)
    description_embedding = text_to_vector(filtered_description)
    
    entity_embeddings = {}
    for entity in entities:
        entity_text = create_text_representation(entity)
        entity_embeddings[entity['name']] = text_to_vector(entity_text)
    
    similarities = {
        name: cosine_similarity(description_embedding.reshape(1, -1), vec.reshape(1, -1))[0][0]
        for name, vec in entity_embeddings.items()
    }
    
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches[:2]  # Return top 2 matches

def process_dataframe(df, categories, subcategory_mapping):
    """Process DataFrame: classify each row's crime description into categories and subcategories."""
    df['Top_Category_1'] = ""
    df['Top_Score_1'] = 0.0
    df['Top_Category_2'] = ""
    df['Top_Score_2'] = 0.0
    df['Top_Subcategory'] = ""
    df['Subcategory_Score'] = 0.0
    
    for index, row in df.iterrows():
        description = row['crime_description']
        top_2_matches = classify_description(description, categories)
        
        if top_2_matches:
            df.at[index, 'Top_Category_1'] = top_2_matches[0][0]
            df.at[index, 'Top_Score_1'] = top_2_matches[0][1]
        
        if len(top_2_matches) > 1:
            df.at[index, 'Top_Category_2'] = top_2_matches[1][0]
            df.at[index, 'Top_Score_2'] = top_2_matches[1][1]
        
        # Perform subcategory classification
        top_category = df.at[index, 'Top_Category_1']
        subcategory_json = subcategory_mapping.get(top_category)
        
        if subcategory_json:
            subcategories = load_json(subcategory_json)['subcategories']
            top_subcategory_match = classify_description(description, subcategories)
            
            if top_subcategory_match:
                df.at[index, 'Top_Subcategory'] = top_subcategory_match[0][0]
                df.at[index, 'Subcategory_Score'] = top_subcategory_match[0][1]
    
    return df

if __name__ == "__main__":
    # Initialize embeddings
    glove_path = "/content/glove.6B.100d.txt"
    initialize_embeddings(glove_path)
    
    # Load category data
    categories_json = "C:/Users/hp/Documents/GitHub/IndiaAI CyberGaurd Hackathon/SEMANTIC/categories.json"
    categories = load_json(categories_json)['categories']
    
    # Define subcategory JSON mappings
    subcategory_mapping = {
        "Cyber Fraud": "C:/path/to/CyberFraud.json",
        "Online Harassment": "C:/path/to/OnlineHarassment.json",
        "Data Breach": "C:/path/to/DataBreach.json",
        "Illegal Content": "C:/path/to/IllegalContent.json"
    }
    
    # Prepare embedding matrix with all category descriptions
    all_texts = [create_text_representation(cat) for cat in categories]
    prepare_embedding_matrix(all_texts)
    
    # Load dataset
    df = pd.read_csv("your_data.csv")
    
    # Process DataFrame with hierarchical classification
    df = process_dataframe(df, categories, subcategory_mapping)
    
    # Save results
    df.to_csv("classified_crimes_with_subcategories.csv", index=False)
    
    print("Classification completed! Results saved to 'classified_crimes_with_subcategories.csv'.")
