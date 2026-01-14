"""
Classification Demo - K-NN Classifier
Demonstrates K-NN classification using trained model
"""

import sys
import os
import pickle

def demo_classification():
    """Demonstrate K-NN classification using trained model"""
    
    print("=" * 70)
    print("K-NN CLASSIFICATION DEMO")
    print("=" * 70)
    
    # Load trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'data', 'model', 'model_intent_knn.pkl')
    
    if not os.path.exists(model_path):
        print("⚠️  Model not found. Please train the model first.")
        print(f"Expected path: {model_path}")
        return
    
    print(f"\n[1/3] Loading trained K-NN model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✅ Model loaded successfully!")
    print(f"   • Training samples: {len(model_data.get('X_train', []))} documents")
    print(f"   • Classes: {set(model_data.get('y_train', []))}")
    
    # Test queries
    test_queries = [
        "bagaimana cara daftar kuliah",
        "kapan bayar ukt semester ini",
        "jadwal kuliah hari ini apa saja",
        "info beasiswa untuk mahasiswa"
    ]
    
    print("\n[2/3] Testing classifier with sample queries...")
    print("-" * 70)
    
    # Import predict function from training_logic
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chatbot'))
    try:
        from bot_app.training_logic import prediksi_niat_user
        
        for query in test_queries:
            prediction = prediksi_niat_user(query)
            print(f"Query: {query}")
            print(f"   → Predicted Intent: {prediction}\n")
        
        print("[3/3] ✅ K-NN Classification Demo Complete!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"⚠️  Could not import prediction function: {e}")
        print("Demo can only show loaded model information.")

if __name__ == "__main__":
    demo_classification()
