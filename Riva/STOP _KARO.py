from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
from nltk.corpus import stopwords
import re

# Initialize stopwords
stop_words = set(stopwords.words('english'))
indic_stopwords = set([
    "aur", "ki", "hai", "huyi", "ho", "mein", "ye", "ke", "jo", "saath", "ko",
    "bhi", "tatha", "par", "se", "kisi", "un", "apna", "tum", "main", "aap", "inhe",
    "in", "abhi", "ab", "woh", "hum", "unka", "is", "us", "kintu", "athva", "nahin",
    "kar","firto","fir","kese","esse","ka","kabhi","karna"
])
stop_words = stop_words.union(indic_stopwords)

def remove_stopwords(text):
    """Remove stopwords from the given text."""
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords and join back
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def load_categories(json_file):
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data['categories']

def create_category_descriptions(category):
    # Get descriptions and remove stopwords
    description = remove_stopwords(category.get('description', ''))
    transliterated_description = remove_stopwords(category.get('transliterated_description', ''))
    
    # Process examples
    examples = category.get('related_examples', [])
    # Remove stopwords from each example
    filtered_examples = [remove_stopwords(example) for example in examples]
    # Join examples with a separator and repeat them
    related_examples = " | ".join(filtered_examples) + " " + " ".join(filtered_examples)
    
    return f"{description} {transliterated_description} {related_examples}".strip()

def classify_crime(description, categories):
    # Remove stopwords from the input description
    filtered_description = remove_stopwords(description)
    
    # Load a pre-trained Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create comprehensive descriptions for each category
    category_descriptions = {
        cat['name']: create_category_descriptions(cat)
        for cat in categories
    }
    
    # Encode the filtered crime description
    description_embedding = model.encode(filtered_description, convert_to_tensor=True)
    
    # Encode filtered category descriptions
    category_embeddings = {
        cat: model.encode(desc, convert_to_tensor=True)
        for cat, desc in category_descriptions.items()
    }
    
    # Compute cosine similarities
    similarities = {
        cat: util.pytorch_cos_sim(description_embedding, emb)[0].item()
        for cat, emb in category_embeddings.items()
    }
    
    # Sort categories by similarity score
    sorted_matches = sorted(
        similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get top 3 matches
    top_3_matches = sorted_matches[:3]
    
    # Find the category with highest similarity
    best_match = top_3_matches[0][0]
    
    return best_match, similarities, top_3_matches

def explain_classification(top_matches, categories):
    explanation = []
    for cat_name, score in top_matches:
        # Find the category definition safely
        category = next(cat for cat in categories if cat['name'] == cat_name)
        
        explanation.append(f"\n{cat_name} (Similarity: {score:.4f}):")
        explanation.append(f"- Description: {category.get('description', 'N/A')}")
        
        # Handle transliterated description safely
        transliterated_desc = category.get('transliterated_description', None)
        if transliterated_desc:
            explanation.append(f"- Hindi Transliteration: {transliterated_desc}")
        
        # Add related examples if available
        examples = category.get('related_examples', None)
        if examples:
            explanation.append(f"- Examples: {', '.join(examples)}")
    
    return '\n'.join(explanation)

# Example usage
if __name__ == "__main__":
    # Load the categories from your JSON file
    with open(r"C:\Users\hp\Documents\GitHub\IndiaAI CyberGaurd Hackathon\SEMANTIC\categories.json", "r", encoding="utf-8") as f:
        categories_data = json.load(f)
        categories = categories_data.get('categories', [])

    # Example crime description
    crime_description = """
    m ek aadmi hoon aur mujhe ek ladki zabardasti ghar le ja rahi hai
    """

    # Classify the crime
    best_match, all_similarities, top_3 = classify_crime(crime_description, categories)

    print("\nBest matching category:", best_match)
    print("\nTop 3 matching categories with explanation:")
    print(explain_classification(top_3, categories))

    print("\nAll similarity scores:")
    for cat, score in sorted(all_similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat}: {score:.4f}")