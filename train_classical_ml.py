import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from preprocessing import clean_text

# Load merged data
df = pd.read_csv('data/fake_or_real_news.csv')

print("="*60)
print("DATASET INFO")
print("="*60)
print(f"Dataset columns: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nFirst row:")
print(df.iloc[0])

# Check if label column exists
if 'label' not in df.columns:
    print("\n✗ ERROR: 'label' column not found!")
    print("Run merge_datasets.py first to combine fake.csv and real.csv")
    exit(1)

# Combine title and text
df['text'] = df['title'] + " " + df['text']

# Clean text
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)
print("Cleaning text...")
df['text_clean'] = df['text'].apply(clean_text)

# Prepare features and target
X = df['text_clean']

# Check for NaN values
nan_count = X.isna().sum()
print(f"NaN values in text: {nan_count}")
X = X.fillna("")

# Convert label to numeric
y = df['label'].map({'FAKE': 1, 'REAL': 0})

print(f"✓ Text cleaned ({len(X)} samples)")

# Split data
print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train label distribution: {y_train.value_counts().to_dict()}")
print(f"Test label distribution: {y_test.value_counts().to_dict()}")

# TF-IDF Features
print("\n" + "="*60)
print("FEATURE EXTRACTION (TF-IDF)")
print("="*60)
print("Extracting TF-IDF features (max_features=5000, bigrams)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Number of features (n-grams): {len(tfidf.get_feature_names_out())}")

# Train Model
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)
print("Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, solver='lbfgs')
clf.fit(X_train_tfidf, y_train)
print("✓ Model trained")

# Predictions
print("\n" + "="*60)
print("EVALUATION")
print("="*60)
y_pred = clf.predict(X_test_tfidf)
y_proba = clf.predict_proba(X_test_tfidf)

# Results
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nInterpretation:")
print(f"  True Negatives (Real correctly identified): {cm[0, 0]}")
print(f"  False Positives (Real wrongly marked Fake): {cm[0, 1]}")
print(f"  False Negatives (Fake wrongly marked Real): {cm[1, 0]}")
print(f"  True Positives (Fake correctly identified): {cm[1, 1]}")

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(clf, 'logreg.pkl')
print("✓ Saved: tfidf.pkl")
print("✓ Saved: logreg.pkl")

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
print("Model trained and saved. You can now use api.py to make predictions.")
