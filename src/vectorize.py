"""
Vectorization module for STKI
Handles TF-IDF vectorization for document representation
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_tfidf_vectorizer(documents, max_features=1000):
    """
    Create TF-IDF vectorizer from documents
    
    Args:
        documents: List of preprocessed documents
        max_features: Maximum number of features
        
    Returns:
        vectorizer: Fitted TfidfVectorizer
        vectors: Document vectors
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(documents)
    return vectorizer, vectors

def save_vectorizer(vectorizer, filepath):
    """Save vectorizer to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(filepath):
    """Load vectorizer from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Example usage
    docs = ["dokumen pertama", "dokumen kedua", "dokumen ketiga"]
    vec, vecs = create_tfidf_vectorizer(docs)
    print(f"Vocabulary size: {len(vec.vocabulary_)}")
    print(f"Document vectors shape: {vecs.shape}")
