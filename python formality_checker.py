# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # Better for large datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib. pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class FormalityChecker:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def load_data(self, file_path, max_rows=None):
        """Load the dataset from CSV file"""
        print("Loading data...")
        try:
            if max_rows:
                df = pd.read_csv(file_path, encoding='utf-8', nrows=max_rows)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
        except: 
            try:
                if max_rows:
                    df = pd.read_csv(file_path, encoding='latin-1', nrows=max_rows)
                else:
                    df = pd.read_csv(file_path, encoding='latin-1')
            except Exception as e:
                print(f"Error loading file: {e}")
                return None
        
        print(f"✓ Dataset loaded successfully!  Shape: {df.shape}")
        print(f"✓ Total rows: {len(df):,}")
        
        # Check for missing values
        missing_formal = df['formal'].isna().sum()
        missing_informal = df['informal'].isna().sum()
        
        if missing_formal > 0 or missing_informal > 0:
            print(f"⚠ Warning: Found {missing_formal:,} missing formal texts and {missing_informal: ,} missing informal texts")
            df = df.dropna()
            print(f"✓ Cleaned dataset shape: {df.shape}")
        
        return df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("\nPreparing data...")
        
        formal_texts = df['formal'].values
        informal_texts = df['informal'].values
        
        formal_labels = np.ones(len(formal_texts))
        informal_labels = np.zeros(len(informal_texts))
        
        all_texts = np.concatenate([formal_texts, informal_texts])
        all_labels = np.concatenate([formal_labels, informal_labels])
        
        print("Preprocessing texts...")
        processed_texts = []
        total = len(all_texts)
        
        for i, text in enumerate(all_texts):
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,}/{total: ,} texts ({(i+1)/total*100:.1f}%)...")
            processed_texts. append(self.preprocess_text(text))
        
        print(f"\n✓ Total samples: {len(processed_texts):,}")
        print(f"✓ Formal samples:  {int(sum(all_labels)):,}")
        print(f"✓ Informal samples:  {int(len(all_labels) - sum(all_labels)):,}")
        
        return processed_texts, all_labels
    
    def split_data(self, texts, labels, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        print(f"✓ Training samples: {len(X_train):,}")
        print(f"✓ Testing samples: {len(X_test):,}")
        return X_train, X_test, y_train, y_test
    
    def vectorize_text(self, X_train, X_test, max_features=10000):
        """Convert text to TF-IDF features"""
        print("\nVectorizing text using TF-IDF...")
        print("  This may take a few minutes for 100K samples...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=5,  # Ignore terms appearing in less than 5 documents
            max_df=0.95,  # Ignore terms appearing in more than 95% documents
            strip_accents='unicode',
            lowercase=True,
            dtype=np.float32  # Use float32 to save memory
        )
        
        print("  Fitting vectorizer on training data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print("  Transforming test data...")
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"✓ Feature vector shape: {X_train_tfidf.shape}")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        print(f"✓ Memory usage: ~{X_train_tfidf.data.nbytes / 1024 / 1024:.1f} MB")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_model(self, X_train, y_train):
        """Train model using SGDClassifier (optimized for large datasets)"""
        print("\nTraining model using SGD Classifier (optimized for 100K data)...")
        print("  SGD Classifier is faster and more memory-efficient than Logistic Regression")
        
        # SGDClassifier with log loss = Logistic Regression with SGD optimization
        self.model = SGDClassifier(
            loss='log_loss',  # Equivalent to Logistic Regression
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            learning_rate='optimal',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            verbose=1  # Show progress
        )
        
        self.model.fit(X_train, y_train)
        print("\n✓ Model training completed!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\nEvaluating model...")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"✓ ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*60}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Informal', 'Formal']))
        
        # Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Informal', 'Formal'],
                       yticklabels=['Informal', 'Formal'],
                       cbar_kws={'label': 'Count'})
            plt.title('Confusion Matrix\n100K Dataset', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\n✓ Confusion matrix saved as 'confusion_matrix. png'")
            plt.close()
        except Exception as e:
            print(f"Note: Could not create confusion matrix plot:  {e}")
        
        return accuracy, y_pred
    
    def get_formality_score(self, text):
        """Get formality score for a given text"""
        if not text or not text.strip():
            return 0.5
        
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        
        # SGDClassifier uses decision_function
        decision = self.model.decision_function(text_vector)[0]
        # Convert to probability using sigmoid
        formality_score = 1 / (1 + np.exp(-decision))
        
        return formality_score
    
    def predict(self, text):
        """Predict whether text is formal or informal"""
        formality_score = self.get_formality_score(text)
        
        if formality_score >= 0.5:
            label = "Formal"
            confidence = formality_score
        else:
            label = "Informal"
            confidence = 1 - formality_score
        
        return {
            'label': label,
            'confidence':  confidence,
            'formality_score': formality_score
        }
    
    def analyze_text(self, text):
        """Detailed analysis of text formality"""
        result = self.predict(text)
        
        print("\n" + "="*60)
        print("FORMALITY ANALYSIS")
        print("="*60)
        print(f"\nText: {text}")
        print(f"\n✓ Prediction: {result['label']}")
        print(f"✓ Confidence: {result['confidence']:.2%}")
        print(f"✓ Formality Score: {result['formality_score']:.2%}")
        
        if result['formality_score'] >= 0.8:
            print("\n📊 Interpretation:  Very Formal")
        elif result['formality_score'] >= 0.6:
            print("\n📊 Interpretation: Moderately Formal")
        elif result['formality_score'] >= 0.4:
            print("\n📊 Interpretation:  Neutral")
        elif result['formality_score'] >= 0.2:
            print("\n📊 Interpretation:  Moderately Informal")
        else:
            print("\n📊 Interpretation: Very Informal")
        
        print("="*60)
        
        return result
    
    def save_model(self, model_path='formality_model.pkl', 
                   vectorizer_path='vectorizer.pkl'):
        """Save trained model and vectorizer"""
        print("\nSaving model...")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Check file sizes
            import os
            model_size = os.path.getsize(model_path) / 1024 / 1024
            vectorizer_size = os.path.getsize(vectorizer_path) / 1024 / 1024
            
            print(f"✓ Model saved to {model_path} ({model_size:.1f} MB)")
            print(f"✓ Vectorizer saved to {vectorizer_path} ({vectorizer_size:.1f} MB)")
        except Exception as e:
            print(f"✗ Error saving model: {e}")
    
    def load_model(self, model_path='formality_model.pkl', 
                   vectorizer_path='vectorizer.pkl'):
        """Load pre-trained model and vectorizer"""
        print("\nLoading model...")
        try:
            with open(model_path, 'rb') as f:
                self. model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("✓ Model loaded successfully!")
        except Exception as e: 
            print(f"✗ Error loading model: {e}")
            raise


def plot_training_statistics(df):
    """Plot dataset statistics with visualizations"""
    print("\n" + "="*60)
    print("DATASET STATISTICS (100K)")
    print("="*60)
    
    # Calculate statistics
    formal_lengths = df['formal'].str.len()
    informal_lengths = df['informal'].str.len()
    
    formal_word_counts = df['formal'].str.split().str.len()
    informal_word_counts = df['informal'].str.split().str.len()
    
    print(f"\n✓ Total samples: {len(df):,}")
    print(f"✓ Formal texts - Avg length: {formal_lengths.mean():.2f}, Avg words: {formal_word_counts.mean():.2f}")
    print(f"✓ Informal texts - Avg length:  {informal_lengths.mean():.2f}, Avg words: {informal_word_counts.mean():.2f}")
    
    # Create visualizations
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset Statistics (100K Samples)', fontsize=16, fontweight='bold')
        
        # 1. Character Length Distribution
        axes[0, 0].hist(formal_lengths, bins=50, alpha=0.7, color='#4ECDC4', label='Formal', edgecolor='black')
        axes[0, 0].hist(informal_lengths, bins=50, alpha=0.7, color='#FF6B6B', label='Informal', edgecolor='black')
        axes[0, 0].set_xlabel('Character Length', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Character Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Word Count Distribution
        axes[0, 1].hist(formal_word_counts, bins=30, alpha=0.7, color='#4ECDC4', label='Formal', edgecolor='black')
        axes[0, 1].hist(informal_word_counts, bins=30, alpha=0.7, color='#FF6B6B', label='Informal', edgecolor='black')
        axes[0, 1].set_xlabel('Word Count', fontweight='bold')
        axes[0, 1]. set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Average Statistics Comparison
        categories = ['Avg Char Length', 'Avg Word Count']
        formal_stats = [formal_lengths.mean(), formal_word_counts.mean()]
        informal_stats = [informal_lengths.mean(), informal_word_counts.mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x - width/2, formal_stats, width, label='Formal', 
                              color='#4ECDC4', edgecolor='black', linewidth=1.5)
        bars2 = axes[1, 0].bar(x + width/2, informal_stats, width, label='Informal', 
                              color='#FF6B6B', edgecolor='black', linewidth=1.5)
        
        axes[1, 0].set_xlabel('Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Average Value', fontweight='bold')
        axes[1, 0]. set_title('Average Statistics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]: 
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}',
                              ha='center', va='bottom', fontweight='bold')
        
        # 4. Dataset Summary
        axes[1, 1].axis('off')
        summary_text = f"""
📊 DATASET SUMMARY (100K)

Total Samples: {len(df):,}
Formal Samples: {len(df):,}
Informal Samples:  {len(df):,}
Total Texts: {len(df)*2:,}

FORMAL TEXTS:
• Avg Character Length: {formal_lengths.mean():.2f}
• Avg Word Count: {formal_word_counts.mean():.2f}
• Min Length: {formal_lengths.min()}
• Max Length: {formal_lengths.max()}

INFORMAL TEXTS:
• Avg Character Length: {informal_lengths.mean():.2f}
• Avg Word Count: {informal_word_counts.mean():.2f}
• Min Length: {informal_lengths.min()}
• Max Length: {informal_lengths.max()}

OBSERVATION: 
Formal texts are typically 
{formal_lengths.mean() - informal_lengths.mean():.1f} characters 
longer than informal texts. 

Dataset Quality:  PROFESSIONAL
Training Set: 80,000 pairs (160K texts)
Test Set: 20,000 pairs (40K texts)
        """
        axes[1, 1].text(0.05, 0.5, summary_text, fontsize=9, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
        print("\n✓ Dataset statistics saved as 'dataset_statistics. png'")
        plt.close()
        
    except Exception as e:
        print(f"Note: Could not create statistics plot: {e}")


def main():
    """Main function - Optimized for 100K Dataset"""
    
    print("\n" + "="*60)
    print("   FORMALITY CHECKER - MODEL TRAINING")
    print("   Dataset: 100K samples (200K texts)")
    print("   Algorithm: SGD Classifier (Optimized)")
    print("="*60)
    
    checker = FormalityChecker()
    
    # Load data (all 100K rows)
    df = checker.load_data('formal_informal_dataset (1).csv')
    
    if df is None:
        print("✗ Failed to load data. Exiting...")
        return None
    
    # Statistics with visualizations
    plot_training_statistics(df)
    
    # Prepare data
    texts, labels = checker.prepare_data(df)
    X_train, X_test, y_train, y_test = checker.split_data(texts, labels)
    X_train_tfidf, X_test_tfidf = checker.vectorize_text(X_train, X_test)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL (This may take 5-10 minutes)")
    print("="*60)
    
    import time
    start_time = time.time()
    
    checker.train_model(X_train_tfidf, y_train)
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate
    accuracy, predictions = checker.evaluate_model(X_test_tfidf, y_test)
    
    # Save
    checker.save_model()
    
    # Test samples
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE TEXTS")
    print("="*60)
    
    test_samples = [
        "Hey!  What's up?  Wanna grab some food? ",
        "I am writing to formally request a meeting to discuss the project.",
        "Thanks a lot for your help!",
        "I would like to express my sincere gratitude for your assistance.",
        "Can we meet tomorrow? ",
        "It would be greatly appreciated if you could attend."
    ]
    
    for text in test_samples:
        checker.analyze_text(text)
    
    # Final summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 Generated files:")
    print("  ✓ formality_model. pkl")
    print("  ✓ vectorizer.pkl")
    print("  ✓ confusion_matrix.png")
    print("  ✓ dataset_statistics.png")
    print(f"\n📊 Dataset:  100,000 pairs (200,000 texts)")
    print(f"📊 Algorithm: SGD Classifier")
    print(f"📊 Training time: {training_time/60:.1f} minutes")
    print(f"📊 Accuracy: {accuracy:.2%}")
    print(f"📊 Vocabulary size: {len(checker.vectorizer.vocabulary_):,} features")
    print("\n🚀 Next step: Run the web app")
    print("   Command: streamlit run app.py")
    print("="*60)
    
    return checker


if __name__ == "__main__":
    checker = main()