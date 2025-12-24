import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction. text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
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
        import os
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024 / 1024
            print(f"✓ File size: {file_size:.2f} MB")
        missing_formal = df['formal']. isna().sum()
        missing_informal = df['informal'].isna().sum()
        if missing_formal > 0 or missing_informal > 0:
            print(f"⚠ Warning: Found {missing_formal: ,} missing formal texts and {missing_informal:,} missing informal texts")
            df = df.dropna()
            print(f"✓ Cleaned dataset shape:  {df.shape}")
        return df
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = ' '.join(text.split())
        return text
    def prepare_data(self, df):
        print("\nPreparing data...")
        formal_texts = df['formal'].values
        informal_texts = df['informal'].values
        formal_labels = np.ones(len(formal_texts))
        informal_labels = np.zeros(len(informal_texts))
        all_texts = np.concatenate([formal_texts, informal_texts])
        all_labels = np. concatenate([formal_labels, informal_labels])
        print("Preprocessing texts...")
        processed_texts = []
        total = len(all_texts)
        for i, text in enumerate(all_texts):
            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  Processed {i+1:,}/{total:,} texts ({(i+1)/total*100:.1f}%)...")
            processed_texts.append(self.preprocess_text(text))
        print(f"\n✓ Total samples: {len(processed_texts):,}")
        print(f"✓ Formal samples: {int(sum(all_labels)):,}")
        print(f"✓ Informal samples: {int(len(all_labels) - sum(all_labels)):,}")
        return processed_texts, all_labels
    def split_data(self, texts, labels, test_size=0.2, random_state=42):
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        print(f"✓ Training samples: {len(X_train):,}")
        print(f"✓ Testing samples: {len(X_test):,}")
        print(f"✓ Train/Test split: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
        return X_train, X_test, y_train, y_test
    def vectorize_text(self, X_train, X_test, max_features=3000):
        print("\nVectorizing text using TF-IDF...")
        print(f"  Target vocabulary size:  {max_features: ,} features")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  
            min_df=2, 
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            dtype=np.float32
        )
        print("  Fitting vectorizer on training data...")
        X_train_tfidf = self.vectorizer. fit_transform(X_train)
        print("  Transforming test data...")
        X_test_tfidf = self.vectorizer.transform(X_test)
        print(f"✓ Feature vector shape: {X_train_tfidf.shape}")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        print(f"✓ Memory usage: ~{X_train_tfidf.data.nbytes / 1024 / 1024:.2f} MB")  # ✅
        vocab_size = len(self.vectorizer.vocabulary_)
        if vocab_size < 500:
            print(f"⚠️  WARNING: Very small vocabulary ({vocab_size})")
            print(f"   Your dataset might be very repetitive")
        elif vocab_size > 2500:
            print(f"✅ Good vocabulary size for {len(X_train):,} samples")    
        return X_train_tfidf, X_test_tfidf
    def train_model(self, X_train, y_train):
        print("\nTraining model using Logistic Regression...")
        print("  Logistic Regression is optimal for datasets < 10,000 samples")
        self. model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0, 
            solver='lbfgs',
            verbose=0
        )
        self.model.fit(X_train, y_train)
        print("✓ Model training completed!")
        return self.model
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        print("\nEvaluating model...")
        print("  Calculating training set performance...")
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print("  Calculating test set performance...")
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"\n{'='*60}")
        print(f"✓ TRAINING ACCURACY: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"✓ TESTING ACCURACY:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"✓ DIFFERENCE: {abs(train_accuracy - test_accuracy):.4f} ({abs(train_accuracy - test_accuracy)*100:.2f}%)")
        print(f"{'='*60}")
        gap = abs(train_accuracy - test_accuracy)
        if gap < 0.03: 
            print("\n✅ Model Status: EXCELLENT - Well-generalized")
        elif gap < 0.05:
            print("\n✅ Model Status: GOOD - Acceptable performance")
        elif gap < 0.10:
            print("\n⚠️  Model Status: FAIR - Slight overfitting")
        else:
            print("\n❌ Model Status: POOR - Significant overfitting")
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT (Test Set)")
        print("="*60)
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Informal', 'Formal']))
        try:
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Informal', 'Formal'],
                       yticklabels=['Informal', 'Formal'],
                       cbar_kws={'label': 'Count'},
                       linewidths=2, linecolor='black',
                       annot_kws={'fontsize':  16, 'fontweight': 'bold'})    
            plt.title(f'Confusion Matrix - Test Set\nTest Accuracy: {test_accuracy:. 2%} | Train Accuracy: {train_accuracy:. 2%}', 
                     fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=13, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
            plt.text(0.5, -0.15, 
                    f'Training:  {train_accuracy:.2%} | Testing: {test_accuracy:. 2%} | Gap: {gap:.2%}',
                    ha='center', transform=plt.gca().transAxes, 
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\n✓ Confusion matrix saved as 'confusion_matrix. png'")
            plt.close()
        except Exception as e:
            print(f"Note: Could not create confusion matrix plot:  {e}")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            categories = ['Training\nAccuracy', 'Testing\nAccuracy']
            accuracies = [train_accuracy, test_accuracy]
            colors = ['#4ECDC4', '#FF6B6B']
            bars = axes[0].bar(categories, accuracies, color=colors, edgecolor='black', 
                               linewidth=2, alpha=0.8, width=0.5)
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                            f'{acc:.2%}',
                            ha='center', va='bottom', fontweight='bold', fontsize=14)
            axes[0].set_ylabel('Accuracy', fontweight='bold', fontsize=12)
            axes[0].set_title('Training vs Testing Accuracy\n(1,400 Sample Dataset)', 
                             fontweight='bold', fontsize=14)
            axes[0]. set_ylim([0.85, 1.0])
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics = ['Precision', 'Recall', 'F1-Score']
            train_metrics = [
                precision_score(y_train, y_train_pred),
                recall_score(y_train, y_train_pred),
                f1_score(y_train, y_train_pred)
            ]
            test_metrics = [
                precision_score(y_test, y_test_pred),
                recall_score(y_test, y_test_pred),
                f1_score(y_test, y_test_pred)
            ]
            x = np.arange(len(metrics))
            width = 0.35
            bars1 = axes[1].bar(x - width/2, train_metrics, width, label='Training',
                               color='#4ECDC4', edgecolor='black', linewidth=1.5)
            bars2 = axes[1].bar(x + width/2, test_metrics, width, label='Testing',
                               color='#FF6B6B', edgecolor='black', linewidth=1.5)
            axes[1]. set_ylabel('Score', fontweight='bold', fontsize=12)
            axes[1].set_title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
            axes[1]. set_xticks(x)
            axes[1].set_xticklabels(metrics)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3, linestyle='--')
            axes[1].set_ylim([0.85, 1.0])
            for bars in [bars1, bars2]: 
                for bar in bars:
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.tight_layout()
            plt.savefig('training_vs_testing_performance.png', dpi=300, bbox_inches='tight')
            print("✓ Training vs Testing performance chart saved as 'training_vs_testing_performance.png'")
            plt.close()
        except Exception as e:
            print(f"Note: Could not create performance comparison chart: {e}")
        return train_accuracy, test_accuracy, y_test_pred
    def get_formality_score(self, text):
        if not text or not text.strip():
            return 0.5
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        proba = self.model.predict_proba(text_vector)[0]
        formality_score = proba[1]
        return formality_score
    def predict(self, text):
        formality_score = self.get_formality_score(text)
        if formality_score >= 0.5:
            label = "Formal"
            confidence = formality_score
        else: 
            label = "Informal"
            confidence = 1 - formality_score
        return {
            'label': label,
            'confidence': confidence,
            'formality_score': formality_score
        }
    def analyze_text(self, text):
        result = self.predict(text)
        print("\n" + "="*60)
        print("FORMALITY ANALYSIS")
        print("="*60)
        print(f"\nText: {text}")
        print(f"\n✓ Prediction: {result['label']}")
        print(f"✓ Confidence: {result['confidence']:.2%}")
        print(f"✓ Formality Score: {result['formality_score']:.2%}")
        if result['formality_score'] >= 0.8:
            print("\n📊 Interpretation: Very Formal")
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
        print("\nSaving model...")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            import os
            model_size = os.path.getsize(model_path) / 1024 / 1024
            vectorizer_size = os.path.getsize(vectorizer_path) / 1024 / 1024
            print(f"✓ Model saved to {model_path} ({model_size:.2f} MB)")
            print(f"✓ Vectorizer saved to {vectorizer_path} ({vectorizer_size:.2f} MB)")
        except Exception as e:
            print(f"✗ Error saving model: {e}")
    def load_model(self, model_path='formality_model.pkl', 
                   vectorizer_path='vectorizer. pkl'):
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
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    formal_lengths = df['formal']. str.len()
    informal_lengths = df['informal'].str.len()
    formal_word_counts = df['formal'].str.split().str.len()
    informal_word_counts = df['informal'].str.split().str.len()
    print(f"\n✓ Total samples: {len(df):,}")
    print(f"✓ Formal texts - Avg length: {formal_lengths.mean():.2f}, Avg words: {formal_word_counts.mean():.2f}")
    print(f"✓ Informal texts - Avg length:  {informal_lengths.mean():.2f}, Avg words: {informal_word_counts.mean():.2f}")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Dataset Statistics ({len(df):,} Samples)', fontsize=16, fontweight='bold')
        axes[0, 0].hist(formal_lengths, bins=30, alpha=0.7, color='#4ECDC4', label='Formal', edgecolor='black')
        axes[0, 0].hist(informal_lengths, bins=30, alpha=0.7, color='#FF6B6B', label='Informal', edgecolor='black')
        axes[0, 0].set_xlabel('Character Length', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Character Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 1].hist(formal_word_counts, bins=20, alpha=0.7, color='#4ECDC4', label='Formal', edgecolor='black')
        axes[0, 1].hist(informal_word_counts, bins=20, alpha=0.7, color='#FF6B6B', label='Informal', edgecolor='black')
        axes[0, 1].set_xlabel('Word Count', fontweight='bold')
        axes[0, 1]. set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
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
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}',
                              ha='center', va='bottom', fontweight='bold')
        axes[1, 1].axis('off')
        summary_text = f"""
📊 DATASET SUMMARY

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
• Min Length:  {informal_lengths.min()}
• Max Length: {informal_lengths.max()}

OBSERVATION:  
Formal texts are typically 
{formal_lengths.mean() - informal_lengths.mean():.1f} characters 
longer than informal texts. 

Dataset Quality:  ACADEMIC
Training Set: {int(len(df)*0.8):,} pairs
Test Set: {int(len(df)*0.2):,} pairs
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
    
    print("\n" + "="*60)
    print("   FORMALITY CHECKER - MODEL TRAINING")
    print("   Dataset: ~1,400 samples (~2,800 texts)")
    print("   Algorithm: Logistic Regression")
    print("="*60)
    checker = FormalityChecker()
    df = checker.load_data('formal_informal_dataset (1).csv')
    if df is None:
        print("✗ Failed to load data.  Exiting...")
        return None
    plot_training_statistics(df)
    texts, labels = checker.prepare_data(df)
    X_train, X_test, y_train, y_test = checker.split_data(texts, labels)
    X_train_tfidf, X_test_tfidf = checker.vectorize_text(X_train, X_test)
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    import time
    start_time = time. time()
    checker.train_model(X_train_tfidf, y_train)
    training_time = time.time() - start_time
    if training_time < 60:
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
    else:
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
    train_accuracy, test_accuracy, predictions = checker.evaluate_model(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    checker.save_model()
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE TEXTS")
    print("="*60)
    test_samples = [
        "Hey!  What's up?  Wanna grab some food? ",
        "I am writing to formally request a meeting to discuss the project.",
        "Thanks a lot for your help! ",
        "I would like to express my sincere gratitude for your assistance.",
        "Can we meet tomorrow? ",
        "It would be greatly appreciated if you could attend."
    ]
    for text in test_samples:
        checker.analyze_text(text)
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 Generated files:")
    print("  ✓ formality_model. pkl")
    print("  ✓ vectorizer.pkl")
    print("  ✓ confusion_matrix.png")
    print("  ✓ dataset_statistics.png")
    print("  ✓ training_vs_testing_performance.png")
    print(f"\n📊 Dataset Size: {len(df):,} pairs ({len(df)*2:,} texts)")
    print(f"📊 Algorithm: Logistic Regression")
    if training_time < 60:
        print(f"📊 Training Time: {training_time:.2f} seconds")
    else:
        print(f"📊 Training Time: {training_time/60:.2f} minutes")
    print(f"📊 Training Accuracy: {train_accuracy:.2%}")
    print(f"📊 Testing Accuracy: {test_accuracy:.2%}")
    print(f"📊 Accuracy Gap: {abs(train_accuracy - test_accuracy):.2%}")
    print(f"📊 Vocabulary Size: {len(checker.vectorizer.vocabulary_):,} features")
    gap = abs(train_accuracy - test_accuracy)
    if gap < 0.03:
        print("\n✅ Model Quality: EXCELLENT - Well Generalized")
    elif gap < 0.05:
        print("\n✅ Model Quality:  GOOD - Acceptable Performance")
    else:
        print("\n⚠️  Model Quality:  FAIR - Some Overfitting")
    print("\n🎓 Perfect for Academic Projects!")
    print("🚀 Next step: Run the web app")
    print("   Command: streamlit run app.py")
    print("="*60)
    return checker
if __name__ == "__main__":
    checker = main()