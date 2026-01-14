"""
Sentiment Analysis Inference Module
Load dari sentiment_model.pkl
"""

import torch
import pickle
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PKL = os.path.join(ROOT_DIR, 'model', 'sentimen_analisis.pkl')

class SentimentAnalyzer:
    def __init__(self, model_path=None):
        """Initialize sentiment analyzer from .pkl file"""
        if model_path is None:
            model_path = MODEL_PKL
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 128
        self.id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        
        # Load model if exists
        if os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load model from pickle file"""
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
            print(f"❌ Error loading model: {e}")
            print(f"   Train model first: python train_sentiment_indobert.py")
            return False
    
    def predict(self, text, return_confidence=True):
        """Predict sentiment for given text"""
        if self.model is None or self.tokenizer is None:
            return ("error", {"error": "Model not loaded"}) if return_confidence else "error"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
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


# Testing
if __name__ == "__main__":
    import sys
    
    analyzer = SentimentAnalyzer()
    
    if analyzer.model is None:
        print("\n❌ Model not found!")
        print("Run: python train_sentiment_indobert.py\n")
        sys.exit(1)
    
    test_texts = [
        "Makanannya enak banget! Saya sangat suka.",
        "Harga terlalu mahal untuk kualitas segini.",
        "Tempatnya nyaman, tapi makanannya biasa saja."
    ]
    
    print("\n" + "="*60)
    print("TESTING SENTIMENT ANALYZER")
    print("="*60)
    
    for text in test_texts:
        sentiment, confidence = analyzer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment.upper()}")
        print(f"Confidence: Pos={confidence['positive']:.2%}, " 
              f"Neu={confidence['neutral']:.2%}, "
              f"Neg={confidence['negative']:.2%}")
    
    print("\n" + "="*60)
