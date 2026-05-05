# main.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.optimize import linear_sum_assignment
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
# The project requires us to not just apply PCA/SVD but also to analyze
# how much variance is retained and how that affects downstream performance.
print("--- Reducing Dimensions ---")

# --- TF-IDF: Truncated SVD ---
# TruncatedSVD is used instead of PCA because TF-IDF produces sparse matrices.
# Regular PCA requires dense input and would be extremely memory-intensive here.
svd = TruncatedSVD(n_components=100, random_state=config.RANDOM_STATE)
x_tfidf_reduced = svd.fit_transform(x_tfidf)

# Same reduction for BoW so we can run classifiers on it too
svd_bow = TruncatedSVD(n_components=100, random_state=config.RANDOM_STATE)
x_bow_reduced = svd_bow.fit_transform(x_bow)

# --- BERT: PCA ---
# BERT embeddings are already dense (768-dim), so standard PCA applies here.
pca = PCA(n_components=100, random_state=config.RANDOM_STATE)
x_bert_reduced = pca.fit_transform(x_bert)

# --- Explained Variance Analysis ---
# This is a required part of Part D. We need to show how many components
# are needed to capture most of the variance, and whether 100 is a reasonable cutoff.

def plot_explained_variance(reducer, title, color='steelblue'):
    """
    Plots cumulative explained variance ratio against number of components.
    Draws a horizontal line at 90% as a common practical threshold.
    """
    cumulative_variance = np.cumsum(reducer.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             color=color, linewidth=2)
    plt.axhline(y=0.90, color='red', linestyle='--', label='90% variance threshold')
    plt.axvline(x=100, color='gray', linestyle=':', label='n_components=100')
    plt.fill_between(range(1, len(cumulative_variance) + 1), cumulative_variance, alpha=0.2, color=color)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print the exact variance retained at 100 components so it can go in the report
    variance_at_100 = cumulative_variance[99]
    print(f"  [{title}] Variance retained with 100 components: {variance_at_100:.4f} ({variance_at_100*100:.2f}%)")

print("\n--- Explained Variance Analysis ---")
plot_explained_variance(svd, "TF-IDF: Cumulative Explained Variance (TruncatedSVD)", color='darkorange')
plot_explained_variance(pca, "BERT: Cumulative Explained Variance (PCA)", color='steelblue')

# For BoW SVD as well, since we are evaluating it
variance_bow = np.cumsum(svd_bow.explained_variance_ratio_)
print(f"  [BoW SVD] Variance retained with 100 components: {variance_bow[99]:.4f} ({variance_bow[99]*100:.2f}%)")


# 4. Model Training & Evaluation (Part C)

def map_kmeans_labels(cluster_labels, true_labels, n_clusters):
    """
    KMeans assigns arbitrary cluster IDs (0, 1, 2...) that have no inherent
    correspondence to the true class labels. To evaluate it fairly, we use the
    Hungarian algorithm to find the optimal 1-to-1 mapping between cluster IDs
    and true class IDs that maximizes agreement.
    """
    # Build a confusion-like cost matrix: rows = clusters, cols = true classes
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for cluster_id, true_id in zip(cluster_labels, true_labels):
        cost_matrix[cluster_id][true_id] += 1
    
    # linear_sum_assignment finds the assignment that maximizes total overlap
    # (we negate the matrix because it minimizes by default)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Build the remapping dictionary and apply it
    label_map = {row: col for row, col in zip(row_ind, col_ind)}
    return np.array([label_map[c] for c in cluster_labels])


def evaluate_models(X, y, space_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    n_classes = len(np.unique(y))
    
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "LinearSVM": SVC(kernel='linear'),
        "RBF_SVM": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        # KMeans is unsupervised — it is fit on X_train only (labels are ignored).
        # Cluster labels are then mapped to true classes via the Hungarian algorithm.
        "KMeans": KMeans(n_clusters=n_classes, n_init=10, random_state=config.RANDOM_STATE)
    }
    
    results = []
    for name, model in models.items():
        if name == "KMeans":
            # Unsupervised: fit on training features without labels,
            # then predict on the full test set and remap cluster IDs
            model.fit(X_train)
            raw_preds = model.predict(X_test)
            preds = map_kmeans_labels(raw_preds, y_test, n_classes)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        acc  = accuracy_score(y_test, preds)
        f1   = f1_score(y_test, preds, average='macro')      # Mandatory Metric
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec  = recall_score(y_test, preds, average='macro', zero_division=0)
        
        results.append({
            "Model": name,
            "Space": space_name,
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1, 4),
            "Precision_Macro": round(prec, 4),
            "Recall_Macro": round(rec, 4),
        })
        
        # Print a full per-class breakdown so it can go straight into the report
        print(f"\n  [{space_name}] {name} Classification Report:")
        print(classification_report(y_test, preds, zero_division=0))
    
    return pd.DataFrame(results)

# Run evaluations across all three feature spaces
print("\n--- Evaluating on BoW (SVD-reduced) ---")
results_bow = evaluate_models(x_bow_reduced, y, "BoW (SVD)")

print("\n--- Evaluating on TF-IDF (SVD-reduced) ---")
results_tfidf = evaluate_models(x_tfidf_reduced, y, "TF-IDF (SVD)")

print("\n--- Evaluating on BERT (PCA-reduced) ---")
results_bert = evaluate_models(x_bert_reduced, y, "BERT (PCA)")

# 5. Comparison Table
final_results = pd.concat([results_bow, results_tfidf, results_bert], ignore_index=True)
print("\n--- Final Performance Comparison ---")
print(final_results.to_string(index=False))

# 6. Visualization

# Bar chart: Accuracy across all spaces and models
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

final_results.pivot(index='Model', columns='Space', values='Accuracy').plot(
    kind='bar', ax=axes[0], colormap='tab10'
)
axes[0].set_title("Model Accuracy: BoW vs TF-IDF vs BERT")
axes[0].set_ylabel("Accuracy Score")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
axes[0].legend(loc='lower right')
axes[0].grid(axis='y', alpha=0.3)

# Bar chart: Macro F1 — this is the mandatory metric so it deserves its own panel
final_results.pivot(index='Model', columns='Space', values='F1_Macro').plot(
    kind='bar', ax=axes[1], colormap='tab10'
)
axes[1].set_title("Macro F1 Score: BoW vs TF-IDF vs BERT")
axes[1].set_ylabel("Macro F1 Score")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
axes[1].legend(loc='lower right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Heatmap of F1 scores — gives a cleaner view for the report
pivot_f1 = final_results.pivot(index='Model', columns='Space', values='F1_Macro')
plt.figure(figsize=(8, 5))
import seaborn as sns
sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5)
plt.title("Macro F1 Score Heatmap: Models vs Feature Spaces")
plt.tight_layout()
plt.show()