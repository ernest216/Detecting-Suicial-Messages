import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print(nltk.data.path)


print(stopwords.words('english')[:10])  # Should print a list of English stopwords
print(word_tokenize("This is a test sentence."))  # Should print ['This', 'is', 'a', 'test', 'sentence', '.']
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running"))  # Should print 'run'

# Load dataset
data = pd.read_csv('Suicide_Detection.csv')

# Check the first few rows
print(data.head())

# Check for null values
print(data.isnull().sum())

# Inspect class distribution
print(data['class'].value_counts())


# CountVectorizer for word frequencies
vectorizer = CountVectorizer(stop_words='english', max_features=20)
word_counts = vectorizer.fit_transform(data['text'])
words = vectorizer.get_feature_names_out()

# Separate by class
suicide_text = data[data['class'] == 'suicide']['text']
non_suicide_text = data[data['class'] == 'non-suicide']['text']

suicide_counts = vectorizer.fit_transform(suicide_text).toarray().sum(axis=0)
non_suicide_counts = vectorizer.fit_transform(non_suicide_text).toarray().sum(axis=0)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_text(text):
    # Remove URLs and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Split dataset into features and target
X = data['text']
y = data['class']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert text to numerical representation using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Adjust max_features if needed
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'fontsize': 15})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# Get feature names and their coefficients
feature_names = tfidf.get_feature_names_out()
coefs = model.coef_[0]  # Logistic Regression coefficients

# Select top 20 positive and negative correlated features
top_positive_indices = np.argsort(coefs)[-20:]
top_negative_indices = np.argsort(coefs)[:20]

top_features = np.concatenate([feature_names[top_positive_indices], feature_names[top_negative_indices]])
top_coefs = np.concatenate([coefs[top_positive_indices], coefs[top_negative_indices]])

plt.figure(figsize=(10, 6))
sns.barplot(x=top_coefs, y=top_features, palette="coolwarm")
plt.title('Correlation of Top Features with Suicide Detection')
plt.xlabel('Coefficient')
plt.ylabel('Features')
plt.show()
plt.savefig('feature_correlation.png')
plt.savefig('confusion_matrix.png')

feature_names = tfidf.get_feature_names_out()
importance = np.abs(model.coef_[0])
top_indices = np.argsort(importance)[-10:]  # Top 10 important features

plt.barh(range(10), importance[top_indices], color='blue')

from wordcloud import WordCloud

plt.yticks(range(10), [feature_names[i] for i in top_indices])
plt.title('Top 10 TF-IDF Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

suicide_text = " ".join(data[data['class'] == 'suicide']['text'])
non_suicide_text = " ".join(data[data['class'] == 'non-suicide']['text'])

suicide_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(suicide_text)
non_suicide_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(non_suicide_text)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(suicide_wc, interpolation='bilinear')
plt.title('Suicide Posts Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(non_suicide_wc, interpolation='bilinear')
plt.title('Non-Suicide Posts Word Cloud')
plt.axis('off')
plt.show()
plt.savefig('word_cloud_suicide.png')

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label='suicide')
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
plt.savefig('roc_curve.png')


