"""
HVAC Fault Classification Models
Trains ensemble of ML models for fault detection
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import HVACFeatureEngineer

class HVACFaultClassifier:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_engineer = HVACFeatureEngineer()
        self.label_encoder = LabelEncoder()
        self.fault_types = None
        
    def load_and_prepare_data(self, filepath='hvac_data.csv'):
        """Load and prepare data for training"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        
        # Prepare features
        X = self.feature_engineer.prepare_features(df, fit_scaler=True)
        y = df['fault_type']
        
        # Store fault types
        self.fault_types = sorted(y.unique())
        
        # Encode labels for XGBoost
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y, y_encoded
    
    def train_individual_models(self, X_train, y_train, y_train_encoded):
        """Train individual classifiers"""
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        print("Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train_encoded)
        self.models['xgboost'] = xgb_model
        
        print("Training SVM...")
        svm_model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train, y_train)
        self.models['svm'] = svm_model
        
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
    
    def create_ensemble(self, X_train, y_train):
        """Create voting ensemble of best models"""
        print("\nCreating ensemble model...")
        
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost']),
                ('gb', self.models['gradient_boosting'])
            ],
            voting='soft',
            n_jobs=-1
        )
        
        self.ensemble_model.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test, y_test_encoded):
        """Evaluate all models"""
        results = {}
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for name, model in self.models.items():
            if name == 'xgboost':
                y_pred_encoded = model.predict(X_test)
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            else:
                y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"\n{name.upper()} - Accuracy: {accuracy:.4f}")
        
        # Evaluate ensemble
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            results['ensemble'] = ensemble_accuracy
            
            print(f"\nENSEMBLE MODEL - Accuracy: {ensemble_accuracy:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred_ensemble))
            
            # Confusion matrix
            self.plot_confusion_matrix(y_test, y_pred_ensemble)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.fault_types,
                   yticklabels=self.fault_types)
        plt.title('Confusion Matrix - HVAC Fault Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved to 'confusion_matrix.png'")
    
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        rf_model = self.models['random_forest']
        feature_names = self.feature_engineer.get_feature_importance_names()
        
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.title('Top 15 Feature Importances')
        plt.barh(range(15), importances[indices])
        plt.yticks(range(15), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to 'feature_importance.png'")
    
    def train(self, filepath='hvac_data.csv', test_size=0.2):
        """Complete training pipeline"""
        # Load data
        X, y, y_encoded = self.load_and_prepare_data(filepath)
        
        # Split data
        X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y, y_encoded, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train models
        self.train_individual_models(X_train, y_train, y_train_encoded)
        self.create_ensemble(X_train, y_train)
        
        # Evaluate
        results = self.evaluate_models(X_test, y_test, y_test_encoded)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        return results
    
    def save_models(self, prefix='hvac_model'):
        """Save trained models"""
        with open(f'{prefix}_ensemble.pkl', 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        with open(f'{prefix}_feature_engineer.pkl', 'wb') as f:
            pickle.dump(self.feature_engineer, f)
        
        with open(f'{prefix}_fault_types.pkl', 'wb') as f:
            pickle.dump(self.fault_types, f)
        
        with open(f'{prefix}_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\nModels saved with prefix '{prefix}'")
    
    def load_models(self, prefix='hvac_model'):
        """Load trained models"""
        with open(f'{prefix}_ensemble.pkl', 'rb') as f:
            self.ensemble_model = pickle.load(f)
        
        with open(f'{prefix}_feature_engineer.pkl', 'rb') as f:
            self.feature_engineer = pickle.load(f)
        
        with open(f'{prefix}_fault_types.pkl', 'rb') as f:
            self.fault_types = pickle.load(f)
        
        with open(f'{prefix}_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Models loaded successfully")
    
    def predict(self, data):
        """Predict fault type for new data"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Prepare features
        X = self.feature_engineer.prepare_features(data, fit_scaler=False)
        
        # Predict
        prediction = self.ensemble_model.predict(X)[0]
        probabilities = self.ensemble_model.predict_proba(X)[0]
        
        # Get confidence scores for all classes
        prob_dict = {fault: prob for fault, prob in zip(self.fault_types, probabilities)}
        
        return prediction, prob_dict

if __name__ == "__main__":
    classifier = HVACFaultClassifier()
    
    # Train models
    results = classifier.train('hvac_data.csv')
    
    # Save models
    classifier.save_models()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
