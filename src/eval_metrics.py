"""
Evaluation Metrics module for STKI
Provides functions to calculate accuracy, precision, recall, F1, confusion matrix
"""

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy"""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='weighted'):
    """Calculate precision"""
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def calculate_recall(y_true, y_pred, average='weighted'):
    """Calculate recall"""
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def calculate_f1(y_true, y_pred, average='weighted'):
    """Calculate F1 score"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix"""
    return confusion_matrix(y_true, y_pred)

def print_evaluation_report(y_true, y_pred, target_names=None):
    """
    Print comprehensive evaluation report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of class names
    """
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    print(f"\nAccuracy:  {calculate_accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {calculate_precision(y_true, y_pred):.4f}")
    print(f"Recall:    {calculate_recall(y_true, y_pred):.4f}")
    print(f"F1 Score:  {calculate_f1(y_true, y_pred):.4f}")
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = calculate_confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

def calculate_macro_metrics(y_true, y_pred):
    """Calculate macro-averaged metrics"""
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision_macro': calculate_precision(y_true, y_pred, average='macro'),
        'recall_macro': calculate_recall(y_true, y_pred, average='macro'),
        'f1_macro': calculate_f1(y_true, y_pred, average='macro')
    }

if __name__ == "__main__":
    # Example usage
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1, 0, 1, 2]
    
    print_evaluation_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
