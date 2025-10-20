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
![WhatsApp Image 2025-10-20 at 17 06 17_602d97a1](https://github.com/user-attachments/assets/fab918e5-5391-455d-b24e-7cc32fbbfc2a)


### Text Length Analysis  
![WhatsApp Image 2025-10-20 at 17 06 41_f379eb64](https://github.com/user-attachments/assets/de7cdcfb-c0fe-427b-a79c-c21e455e7d08)


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
![WhatsApp Image 2025-10-20 at 17 07 50_a970a080](https://github.com/user-attachments/assets/4eba80dd-69cb-41c0-aa8a-a34cdcbb5ffd)
![WhatsApp Image 2025-10-20 at 17 08 52_daa2394b](https://github.com/user-attachments/assets/2df9192e-3f16-43ed-b504-2448bd8f7a09)
![WhatsApp Image 2025-10-20 at 17 08 14_d66f69a2](https://github.com/user-attachments/assets/b02b4842-d65f-4385-a6f1-27ab78a8a582)

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
![WhatsApp Image 2025-10-20 at 17 09 14_395cd304](https://github.com/user-attachments/assets/dcb248a6-0736-4af1-990d-fe603011d395)


### Training Time Analysis  
![WhatsApp Image 2025-10-20 at 17 10 01_e36e7b57](https://github.com/user-attachments/assets/69120843-1ad1-4780-9e52-378f0bdd7973)

---

## Model Evaluation  

**Best Performing Model:** Logistic Regression  
- Accuracy: 96.56%  
- Training Time: 25.02 seconds  

### Detailed Classification Report  
![WhatsApp Image 2025-10-20 at 17 11 01_2a544066](https://github.com/user-attachments/assets/1a2cc110-55b4-494c-8bc6-acae6826acad)


### Confusion Matrix  
![WhatsApp Image 2025-10-20 at 17 11 16_17cac8d8](https://github.com/user-attachments/assets/67c4ef47-ed28-4fe5-9cec-04f12fd8de17)


### Normalized Confusion Matrix  
![WhatsApp Image 2025-10-20 at 17 11 26_83436c88](https://github.com/user-attachments/assets/5b5e9c0b-b303-473c-9554-6ef8f780bf04)


### Prediction Results  
![Uploading WhatsApp Image 2025-10-20 at 17.12.18_6cadc522.jpg…]()


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
