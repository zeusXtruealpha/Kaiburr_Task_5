# Consumer Complaint Classification  
### By Niranjan Galla

---

## Project Overview  

This project performs multi-class text classification on consumer complaint data to automatically categorize complaints into predefined product categories using machine learning.

---

## Objective  

Classify consumer complaints into one of four categories:

- Credit reporting, repair, or other  
- Debt collection  
- Consumer Loan  
- Mortgage

---

## Dataset Analysis  

**Dataset Statistics**
- Total Complaints: 515,519  
- Categories: 3 product types  
- Data Source: Consumer Complaint Database (data.gov)

### Category Distribution  
![Category Distribution](images/category_distribution.png)

### Text Length Analysis  
![Text Length Analysis](images/text_length_analysis.png)

---

## Data Preprocessing  

### Text Cleaning Steps  
- Lowercasing: Convert all text to lowercase  
- Punctuation Removal: Remove special characters and punctuation  
- Number Removal: Eliminate numerical values  
- Stopword Removal: Filter out common English stopwords  
- Tokenization: Split text into words  
- Text Normalization: Clean and standardize text format  

### Word Cloud Visualization  
![Word Cloud 1](images/wordcloud_category1.png)  
![Word Cloud 2](images/wordcloud_category2.png)

---

## Feature Engineering  

### TF-IDF Vectorization  
- Features: 2,000 most important terms  
- N-grams: Unigrams and bigrams (1, 2)  
- Vocabulary Size: 2,000  
- Sparsity: Optimized for efficient computation  

### Train-Test Split  
- Training Set: 412,415 samples (80%)  
- Testing Set: 103,104 samples (20%)  
- Sampling: Stratified to maintain class distribution  

---

## Model Training and Comparison  

### Models Evaluated  
- Logistic Regression – Fast and interpretable  
- Naive Bayes – Probabilistic classifier  
- Random Forest – Ensemble method  
- Support Vector Machine (SVM) – Maximum margin classifier  

### Performance Comparison  
![Model Comparison](images/model_comparison.png)

### Training Time Analysis  
![Training Time](images/training_time.png)

---

## Model Evaluation  

**Best Performing Model:** Logistic Regression  
- Accuracy: 96.56%  
- Training Time: 25.02 seconds  

### Detailed Classification Report  
![Classification Report](images/classification_report.png)

### Confusion Matrix  
![Confusion Matrix](images/confusion_matrix.png)

### Normalized Confusion Matrix  
![Normalized Confusion Matrix](images/normalized_confusion_matrix.png)

### Prediction Results  
![Prediction Results](images/prediction_results.png)

---

## Model Persistence  

### Saved Components  
All models and preprocessing components are stored for future use:

- `best_model.pkl` – Best performing classifier  
- `all_models.pkl` – All trained models  
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer  
- `label_encoder.pkl` – Label encoder  
- `clean_text_function.pkl` – Text preprocessing function  
- `training_metadata.pkl` – Model metadata  
- `model_results.csv` – Performance metrics  
- `cleaned_data.csv` – Processed dataset  

---

## Quick Prediction Example  

```python
# Load pre-trained models
model, vectorizer, label_encoder, clean_text = quick_load_complaint_classifier()

# Make predictions
category, confidence = predict_complaint_category(
    "Your complaint text here",
    model, vectorizer, label_encoder, clean_text
)
print(category, confidence)
