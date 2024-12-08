# Suicide Detection in Text Data Using Machine Learning

## 1. Introduction

Suicide is a global public health crisis, ranking among the leading causes of death worldwide [2]. Digital platforms, such as Reddit, often serve as spaces where individuals express emotional struggles, including suicidal ideation. Leveraging machine learning to detect these signals provides an opportunity for timely intervention and support. This project explores the application of Logistic Regression, an interpretable and efficient machine learning model, to classify text messages as either suicidal or non-suicidal. By detecting patterns in text, the model aims to assist mental health professionals in prioritizing cases that require immediate attention.

## 2. Research Topic

The primary research question addressed in this project is: **Can machine learning effectively identify suicidal messages?** This question reflects a significant need within mental health support systems, where early detection can have lifesaving implications. Machine learning offers a scalable and efficient approach to analyzing the increasing volume of text data generated on digital platforms.

## 3. Dataset and Preprocessing

### Dataset
The dataset, titled the *"Suicide and Depression Detection Dataset"*, was obtained from [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch). It consists of over 232,000 posts from Reddit’s *SuicideWatch* and *depression* subreddits. Each post is labeled as either "suicide" or "non-suicide," forming a binary classification task.

### Preprocessing
The dataset underwent comprehensive preprocessing to ensure its usability:
- **Text Cleaning**: Removed URLs, special characters, and irrelevant stopwords.
- **Tokenization**: Split text into individual words.
- **Lemmatization**: Normalized words to their base forms (e.g., "running" → "run").
- **TF-IDF Vectorization**: Transformed text into numerical features by assigning importance based on term frequency and distinctiveness.

These steps ensured the dataset was clean and ready for model training.

## 4. Logistic Regression Technique

Logistic Regression was chosen for its interpretability, efficiency, and suitability for binary classification tasks. This model predicts probabilities for binary outcomes by mapping input features to probabilities using the sigmoid function. Coefficients from the Logistic Regression model revealed the most influential terms, such as "suicide," "die," and "help," which strongly contributed to classifying messages as suicidal. Its computational efficiency makes it ideal for large-scale deployment, even with limited resources.

## 5. Dataset Splitting and Standard Procedure

The dataset was split into training and testing subsets using an 80-20 stratified split, ensuring class distribution remained consistent across subsets. Preprocessing and feature extraction were applied exclusively to the training set to prevent data leakage. The TF-IDF vectorizer was fit on the training set and used to transform the test set, ensuring consistency.

### Running Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name
   cd your-repo-name
2. Install dependencies:
```bash
pip install -r requirements.txt

4. 
