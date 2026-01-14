"""
Sentiment Analysis Inference Module for Chatbot
Load dari sentiment_model.pkl
"""

import torch
import pickle
import os

class SentimentAnalyzer:
    def __init__(self, model_path=None):
        """Initialize sentiment analyzer from .pkl file"""
        if model_path is None:
            # Point to existing model: chatbot/bot_app/model/model_sentiment_analisis/sentimen_analisis.pkl
            # __file__ is in bot_app/sentiment_analyzer.py
            # So base_dir = bot_app/, and model is in bot_app/model/
            base_dir = os.path.dirname(os.path.abspath(__file__))  # bot_app/
            model_path = os.path.join(base_dir, 'model', 'model_sentiment_analisis', 'sentimen_analisis.pkl')
        
        print(f"[SENTIMENT] Initializing with path: {model_path}")
        print(f"[SENTIMENT] Path exists: {os.path.exists(model_path)}")
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ID to label mapping
        self.id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        self.max_length = 128
        
        # Load model if exists
        if os.path.exists(model_path):
            print(f"[SENTIMENT] Model file found, loading...")
            self.load_model()
        else:
            print(f"[SENTIMENT] ❌ Model file not found at {model_path}")
    
    def load_model(self):
        """Load trained model from pickle file"""
        try:
            print(f"Loading sentiment model from {self.model_path}...")
            
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.tokenizer = model_package['tokenizer']
            self.max_length = model_package.get('max_length', 128)
            self.id2label = model_package.get('id2label', {0: 'positive', 1: 'neutral', 2: 'negative'})
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Sentiment model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading sentiment model: {e}")
            return False
    
    def predict(self, text, return_confidence=True):
        """
        Predict sentiment for given text
        
        Args:
            text (str): Input text to classify
            return_confidence (bool): If True, return confidence scores
        
        Returns:
            If return_confidence=True: tuple (sentiment_label, confidence_dict)
            If return_confidence=False: sentiment_label (str)
        """
        if self.model is None or self.tokenizer is None:
            return ("error", {"error": "Model not loaded"}) if return_confidence else "error"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probs, dim=1).item()
        sentiment = self.id2label[predicted_class]
        
        if return_confidence:
            confidence = {
                'positive': float(probs[0][0]),
                'neutral': float(probs[0][1]),
                'negative': float(probs[0][2])
            }
            return sentiment, confidence
        else:
            return sentiment
    
    def batch_predict(self, texts):
        """Predict sentiments for multiple texts"""
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        return results


# Global instance (singleton pattern)
_analyzer = None

def get_analyzer():
    """Get or create global sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

def predict_sentiment(text):
    """
    Simple function to predict sentiment
    Returns: (sentiment_label, confidence_dict)
    """
    analyzer = get_analyzer()
    return analyzer.predict(text)


# Testing function
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Makanannya enak banget! Saya sangat suka dengan pelayanannya.",
        "Harga terlalu mahal untuk kualitas segini.",
        "Tempatnya nyaman, tapi makanannya biasa saja."
    ]
    
    print("="*60)
    print("TESTING SENTIMENT ANALYZER")
    print("="*60)
    
    for text in test_texts:
        sentiment, confidence = analyzer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment.upper()}")
        print(f"Confidence: {confidence}")
    
    print("\n" + "="*60)
