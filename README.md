# Problem statement
Implement multiple classification models -Build an interactive Streamlit web application to demonstrate your models.

# Dataset description
The Credit Card Details dataset contains information about customers and their financial and demographic attributes, along with a label indicating credit risk or repayment behavior. The dataset is commonly used for classification tasks to predict whether a customer is likely to be a good or risky credit holder.

# Models used and its metrics
    Model                         Accuracy   AUC       Precision   Recall   F1       MCC
    Logistic Regression           0.8976     0.7280     1.0000     0.0455   0.0870   0.2019
    Decision Tree                 0.9122     0.7709     0.5909     0.5909   0.5909   0.5417
    kNN                           0.8732     0.7618     0.3000     0.1364   0.1875   0.1410
    Gaussian Naive Bayes          0.1756     0.6133     0.1070     0.9091   0.1914  -0.0038
    Random Forest (Ensemble)      0.9366     0.9272     0.8462     0.5000   0.6286   0.6211
    XGBoost (Ensemble)            0.9366     0.7998     0.8000     0.5455   0.6486   0.6288

# Observation about model performance
- Logistic Regression : It is biased toward the majority class and misses many positive cases. It is not suitable when detecting the minority class is important.

- Decision Tree       : It handles nonlinear patterns better and performs reasonably well but may be prone to overfitting.

- KNN                 : It struggles with high-dimensional data and many encoded features, which reduces effectiveness.

- Naive Bayes         : It predicts many positives, leading to high recall but very poor precision. Its independence assumption does not fit this dataset well.

- Random Forest       : It performs very well by reducing overfitting and capturing complex patterns in the data.

- XGBoost             : provides strong predictive performance and handles complex feature interactions effectively.
