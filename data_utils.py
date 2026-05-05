# data_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import config
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_class_wordclouds(df):
    """
    Creates word clouds for each class to satisfy the 
    'Creativity with Plots' requirement.
    """
    print("--- Generating Word Clouds for Each Class ---")
    
    # Mapping for AG News labels (Adjust if using a different dataset)
    class_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for label, name in class_names.items():
        # Combine all cleaned text for this specific class
        class_text = " ".join(df[df['label'] == label]['cleaned_text'])
        
        # Generate the cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(class_text)
        
        # Display in subplot
        axes[label].imshow(wordcloud, interpolation='bilinear')
        axes[label].set_title(f"Top Words: {name}", fontsize=20)
        axes[label].axis('off')

    plt.tight_layout()
    plt.show()

def load_and_preprocess():
    print("--- Loading Dataset from Hugging Face ---")
    dataset = load_dataset(config.DATASET_NAME, split='train')
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # CORRECT WAY TO SHUFFLE IN PANDAS:
    # .sample(frac=1) shuffles the rows, and we take the first 8000
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[:config.SAMPLE_SIZE]
    
    # Basic Cleaning: Part A (Preprocessing)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        return text

    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

def run_eda(df):
    """Generates mandatory EDA visualizations for the report"""
    print("--- Generating EDA Visuals ---")
    plt.figure(figsize=(12, 5))
    
    # Class Distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=df)
    plt.title("Class Distribution (Target Balance)")

    # Text Length Analysis
    plt.subplot(1, 2, 2)
    df['text_len'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    sns.histplot(df['text_len'], bins=30, kde=True)
    plt.title("Distribution of Document Lengths")
    plt.show()

def get_sparse_features(text_series):
    # (1) Unigram Bag-of-Words
    bow_vec = CountVectorizer(stop_words='english', max_features=2000)
    
    #here, we actual do this as a part of our noise reduction

    # (2) Bigram TF-IDF
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)
    #both bigrams and unigrams are use; we take the top 2000 significant words
    x_tfidf = tfidf_vec.fit_transform(text_series)
    
    return x_bow, x_tfidf