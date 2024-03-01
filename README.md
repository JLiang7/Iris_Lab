# Iris_Lab
```
https://share.streamlit.io/jliang7/iris_lab/iris-ml-app.py
```
## Table of Contents

- [About](#about-the-project)
- [Libraries](#libraries)
- [To Run](#to-run)
- [Modeling](#modeling)
- [Insights-and-Conclusion](#insights-and-conclusion)

---

## About The Project
### 
- Quick Mini Python App using Machine Learning  
- Apply Random Forest modeling to predict the Iris flowers’ class labels (setosa, versicolor or virginica)
- Used K-Fold Validation to find the best Accuracy

## Libraries
> **Python**
```
Pandas
Streamlit
Scikit-learn
```
---

## To Run: 
```  
Streamlit run iris-ml-app.py
```
---

## Modeling
**1. Data Preprocessing and Exploratory Data Analysis**
  >- Used Iris dataset from Sklearns
  
**2. Train/Test Split**
  >- Split data to train and test set (25%)
  
**3. Prepare for Modeling**
  >- Transform Data, Standardize using StandardScaler
  
**4. Picking models**
  >- Random Forest Regression as the main Model
  >- XGBoost to compare the Accuracy
  
**5. Model Selection**
  >- Cross-Validation (Kfold)
---
 
## Insights and Conclusion
```
Achieved K-Fold Validation Accuracy of 95.4% with a Standard Deviation of 6.46%
Accuracy Score of 97%
F-1 Accuary of 97%
```
Confusion Matrix 
|    | 0  | 1  | 2  |
|----|----|----|----| 
| 0  | 13 | 0  | 0  |
| 1  | 0  | 16 | 0  |
| 2  | 0  | 1  | 8  |

**Conclusion**

>- Both the Random Forest and XGBoost has achieved similar Accuracy Scores
>- The dataset is too small to come to any conclusion of which model is better
>- Not shown in this proj but the MAE fluxs a lot <br />
    w/ this dataset, converge faster with scaled values but in the process, lose some info
