~author: 

Yours Truly,
Sir Ryan Gabriel The First

# 📊 Data Analysis Practice Repo

Welcome to my **Data Analysis Practice Journey**.
This repo is structured to help me systematically learn and practice **visualization, preprocessing, data mining, and ML/DL interpretation**.

I’ll be using multiple datasets (Titanic, House Prices, MNIST, Credit Card Fraud, Market Basket, etc.) and applying concepts one by one.

---

## 🗂 Folder Structure

```
data_analysis_practice/
│── datasets/          # Raw datasets (Kaggle / UCI / etc.)
│── notebooks/         # Jupyter notebooks
│    ├── 01_visualization.ipynb
│    ├── 02_preprocessing.ipynb
│    ├── 03_data_mining.ipynb
│    ├── 04_modeling.ipynb
│── README.md          # This guide
```

---

# 🎯 Questline

## **Stage 1 — Visualization (EDA)**

👉 Goal: Learn to explore data, understand distributions, and spot patterns.

* [x] Histogram
* [x] Density Plot (KDE)
* [x] Rug Plot
* [x] Cumulative Distribution (CDF)
* [x] Scatterplot
* [x] Pairplot (scatter matrix)
* [x] Bubble chart
* [x] Hexbin plot
* [x] Boxplot
* [x] Strip plot
* [x] Swarm plot
* [x] Bar chart (stacked/grouped)
* [x] Count plot
* [x] Correlation heatmap
* [x] Clustermap
* [x] Line plot (time-series)
* [x] Violin plot
* [x] t-SNE / UMAP embeddings
* [ ] Rolling average plot
* [ ] Autocorrelation plot
* [ ] Spectrogram (for audio/EEG datasets)
* [ ] ROC Curve
* [ ] Precision-Recall Curve
* [ ] Confusion Matrix Heatmap
* [ ] Learning Curve (train vs val metrics)
* [ ] Feature Importance plot
* [ ] SHAP / Permutation plots


---

## **Stage 2 — Preprocessing**

👉 Goal: Clean, transform, and prepare data for ML/DL.

### Data Cleaning

* [ ] Handle missing values (drop, mean, median, KNN)
* [ ] Remove duplicates
* [ ] Detect & treat outliers

### Data Transformation

* [ ] Normalization (Min-Max scaling)
* [ ] Standardization (Z-score)
* [ ] Log transform
* [ ] Box-Cox / Power transform

### Feature Engineering

* [ ] Encode categorical (Label, One-hot, Embeddings)
* [ ] Polynomial features
* [ ] Binning/discretization
* [ ] Group-based scaling

### Dimensionality Reduction

* [ ] PCA
* [ ] t-SNE
* [ ] UMAP
* [ ] LDA (classification-based)

### DL-Specific Preprocessing

* [ ] Signal filtering (low-pass, band-pass, notch)
* [ ] Image denoising (Gaussian, median, wavelet)
* [ ] Image augmentation (flip, crop, rotation, cutout, mixup)
* [ ] Data balancing (SMOTE, oversampling, undersampling)

---

## **Stage 3 — Data Mining**

👉 Goal: Extract hidden patterns & knowledge.

### Association Rule Learning

* [ ] Apriori algorithm
* [ ] FP-Growth

### Clustering

* [ ] K-Means
* [ ] Hierarchical clustering
* [ ] DBSCAN
* [ ] Gaussian Mixture Models

### Sequential Pattern Mining

* [ ] PrefixSpan
* [ ] GSP (Generalized Sequential Pattern mining)

### Anomaly Detection

* [ ] Isolation Forest
* [ ] One-class SVM
* [ ] Autoencoder anomaly detection

### Feature Selection

* [ ] Filter methods (Chi-square, Mutual Information)
* [ ] Wrapper methods (Recursive Feature Elimination - RFE)
* [ ] Embedded methods (Lasso, Decision Trees)

---

## **Stage 4 — Modeling & Interpretation**

👉 Goal: Train simple models & visualize their behavior.

* [ ] Logistic Regression (Titanic dataset)
* [ ] Random Forest (House Prices)
* [ ] CNN (MNIST dataset)
* [ ] Plot ROC, PR, confusion matrix for models
* [ ] Plot learning curves (train vs val loss/accuracy)
* [ ] Visualize feature importance (trees, SHAP)
* [ ] Grad-CAM on CNN (MNIST/CIFAR images)

---

# 🚀 Datasets to Use

* **Titanic** → missing values, categorical encoding, classification
* **House Prices** → regression, feature engineering
* **MNIST / CIFAR** → image preprocessing, CNNs, Grad-CAM
* **Credit Card Fraud** → imbalanced data, anomaly detection
* **Market Basket Data** → association rule mining (Apriori, FP-Growth)

---

# ✅ Progress Tracking

I’ll tick each method off as I implement it in the notebooks.
Goal is to have a **complete practical library** of EDA, preprocessing, and data mining techniques.


