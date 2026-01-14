"""
Feature Selection module for STKI
Implements Chi-square, Mutual Information, and other feature selection methods
"""

from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
import numpy as np

def chi_square_selection(X, y, k=100):
    """
    Select top k features using Chi-square test
    
    Args:
        X: Feature matrix
        y: Labels
        k: Number of top features to select
        
    Returns:
        selector: Fitted SelectKBest with chi2
        X_new: Transformed feature matrix
    """
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return selector, X_new

def mutual_info_selection(X, y, k=100):
    """
    Select top k features using Mutual Information
    
    Args:
        X: Feature matrix
        y: Labels
        k: Number of top features to select
        
    Returns:
        selector: Fitted SelectKBest with mutual_info
        X_new: Transformed feature matrix
    """
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return selector, X_new

def get_feature_scores(X, y, method='chi2'):
    """
    Get feature importance scores
    
    Args:
        X: Feature matrix
        y: Labels
        method: 'chi2' or 'mutual_info'
        
    Returns:
        scores: Feature importance scores
    """
    if method == 'chi2':
        scores, _ = chi2(X, y)
    elif method == 'mutual_info':
        scores = mutual_info_classif(X, y)
    else:
        raise ValueError("Method must be 'chi2' or 'mutual_info'")
    
    return scores

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    
    selector, X_new = chi_square_selection(X, y, k=10)
    print(f"Original shape: {X.shape}")
    print(f"Selected shape: {X_new.shape}")
    print(f"Top 10 feature indices: {selector.get_support(indices=True)}")
