# 👕 Fabric Quality Classifier
## a. Problem statement  

Classify fabric quality into three categories (Low, Medium, High) based on material properties, manufacturing parameters, and inspection characteristics.

##  b. Dataset description

Source: https://www.kaggle.com/datasets/devendrakushwah08/textile-fabric-quality-classification-dataset 

A raw, uncleaned dataset of ~25,000 records for classifying fabric into Low, Medium, or High quality. Designed to test end-to-end data engineering and ML skills.

Key Characteristics:\
Target: fabric_quality (Multi-class)\
Condition: Dirty (contains missing values, duplicates, and noise).\
Predictive Features: Physical metrics (Thread Count, GSM, Tensile Strength ) and categorical attributes (Material, Weave, Finish).

## c. Models used
### Comparison->
| ML Model Name            |   Accuracy |   AUC |   Precision |   Recall |   F1 Score |   MCC |
|:-------------------------|-----------:|------:|------------:|---------:|-----------:|------:|
| Logistic Regression      |      0.95  | 0.981 |       0.95  |    0.95  |      0.95  | 0.925 |
| Decision Tree            |      0.798 | 0.849 |       0.799 |    0.798 |      0.798 | 0.697 |
| kNN                      |      0.812 | 0.928 |       0.815 |    0.812 |      0.813 | 0.718 |
| Naive Bayes              |      0.616 | 0.803 |       0.615 |    0.616 |      0.615 | 0.424 |
| Random Forest (Ensemble) |      0.876 | 0.966 |       0.879 |    0.876 |      0.877 | 0.815 |
| XGBoost (Ensemble)       |      0.906 | 0.976 |       0.909 |    0.906 |      0.907 | 0.86  |

### Observation->
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest accuracy (95%) and consistency, indicating the data likely has clear linear boundaries. |
| **Decision Tree** | Struggled compared to ensemble methods, likely due to overfitting or inability to capture complex patterns (Accuracy: ~80%). |
| **kNN** | While it had a good AUC, its overall accuracy (~81%) suggests it struggled with the noise or high dimensionality in the dataset. |
| **Naive Bayes** | With the lowest accuracy (~61%) and MCC, it likely failed because the assumption that features are independent doesn't hold true here. |
| **Random Forest (Ensemble)**| Significantly improved upon the single Decision Tree (Accuracy: ~87%) and handled noise well, but still trailed Logistic Regression. |
| **XGBoost (Ensemble)** | Came in second place (Accuracy: ~90%), effectively refining errors, but was slightly outperformed by the simpler Logistic Regression. |