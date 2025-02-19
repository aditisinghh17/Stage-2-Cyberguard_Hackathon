from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_texts(text_list, stop_words):
    processed_texts = []
    for text in text_list:
        words = word_tokenize(text.lower())  # Tokenize & lowercase
        filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
        processed_texts.append(" ".join(filtered_words))  # Join words back into a sentence
    return processed_texts

# Example usage
stop_words = set(stopwords.words("english"))  # Load stopwords
texts = [
    "I wish to report a case of cyber fraud and harassment.",
    "There is an anonymous Telegram user blackmailing me.",
    "Someone is spreading an online video of me without my consent."
]

clean_texts = preprocess_texts(texts, stop_words)
print(clean_texts)