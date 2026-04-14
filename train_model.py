import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import json

def train():
    # Load data and drop duplicates
    df = pd.read_csv('heartdisease.csv').drop_duplicates()
    
    # Missing value handling (Median Imputation)
    df.fillna(df.median(), inplace=True)
    
    X = df.drop(columns='target').values
    
    # CRITICAL FIX: The dataset has target inverted (1=Healthy, 0=Disease). 
    # We invert it back so 1=Disease, 0=Healthy to match PulseAI UI logic.
    Y = 1 - df['target'].values
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model with GridSearchCV (addresses Weak Point #3)
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    }
    
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, Y_train)
    lr = grid.best_estimator_
    
    # Evaluate model with comprehensive metrics (addresses Weak Point #1)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    Y_pred = lr.predict(X_test)
    Y_prob = lr.predict_proba(X_test)[:, 1]
    
    metrics_data = {
        "accuracy": round(accuracy_score(Y_test, Y_pred) * 100, 1),
        "precision": round(precision_score(Y_test, Y_pred) * 100, 1),
        "recall": round(recall_score(Y_test, Y_pred) * 100, 1),
        "f1": round(f1_score(Y_test, Y_pred) * 100, 1),
        "roc_auc": round(roc_auc_score(Y_test, Y_prob), 3),
        "best_params": grid.best_params_
    }
    
    # Save artifacts
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(lr, 'model.joblib')
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f)
        
    print(f"Model trained successfully. Accuracy: {metrics_data['accuracy']}% | F1: {metrics_data['f1']}%")
    print(f"Best Parameters: {grid.best_params_}")

if __name__ == '__main__':
    train()
