# main.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

import config
import data_utils
import embedding_utils

# 1. Setup Data
df = data_utils.load_and_preprocess()
data_utils.run_eda(df) # Run EDA first
data_utils.generate_class_wordclouds(df)
# 2. Feature Extraction
x_bow, x_tfidf = data_utils.get_sparse_features(df['cleaned_text'])
x_bert = embedding_utils.get_bert_embeddings(df['cleaned_text'].tolist())
y = df['label'].values

# 3. Dimensionality Reduction (Part D)
print("--- Reducing Dimensions ---")
svd = TruncatedSVD(n_components=100)
x_tfidf_reduced = svd.fit_transform(x_tfidf)

pca = PCA(n_components=100)
x_bert_reduced = pca.fit_transform(x_bert)

# 4. Model Training & Evaluation (Part C)[cite: 2]
def evaluate_models(X, y, space_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "LinearSVM": SVC(kernel='linear'),
        "RBF_SVM": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "KMeans": KMeans(n_clusters=len(np.unique(y)), n_init=10)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro') # Mandatory Metric[cite: 2]
        results.append({"Model": name, "Space": space_name, "Accuracy": acc, "F1_Macro": f1})
    
    return pd.DataFrame(results)

# Run evaluations
results_sparse = evaluate_models(x_tfidf_reduced, y, "TF-IDF (SVD)")
results_dense = evaluate_models(x_bert_reduced, y, "BERT (PCA)")

# 5. Comparison Table[cite: 2]
final_results = pd.concat([results_sparse, results_dense])
print("\n--- Final Performance Comparison ---")
print(final_results)

# 6. Visualization (Creativity with Plots)[cite: 2]
final_results.pivot(index='Model', columns='Space', values='Accuracy').plot(kind='bar')
plt.title("Model Accuracy: TF-IDF vs. BERT")
plt.ylabel("Accuracy Score")
plt.legend(loc='lower right')
plt.show()