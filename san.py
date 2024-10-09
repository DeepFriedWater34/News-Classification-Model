import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

train_file_path = 'C:\Python\Debug\BBC News Train.csv'
test_file_path = 'C:\Python\Debug\BBC News Test.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

X_train_full = train_df['Text']
y_train_full = train_df['Category']

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1))
])

pipeline.fit(X_train, y_train)

y_val_pred = pipeline.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='weighted')
recall = recall_score(y_val, y_val_pred, average='weighted')
f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

X_test = test_df['Text']
test_predictions = pipeline.predict(X_test)

print("\nFirst 5 test predictions:")
print(test_predictions[:5])