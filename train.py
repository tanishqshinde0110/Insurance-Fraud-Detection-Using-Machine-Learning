import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

def train_model(data_path='claims_data.csv'):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Target variable and Features
    X = df.drop(columns=['fraud_reported'])
    y = df['fraud_reported'].map({'Y': 1, 'N': 0})
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # Model Pipeline incorporating SMOTE for class imbalance and XGBoost
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    
    # Split mapping
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model (includes preprocessing + SMOTE + XGBoost)...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model_filename = 'fraud_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"\nModel pipeline saved to {model_filename}")

if __name__ == "__main__":
    train_model()
