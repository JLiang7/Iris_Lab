import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#Page Setup
st.set_page_config(
    page_title = 'Iris Flower Prediction App',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

st.write("""
# Iris Flower Prediction App
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the random forest classifier model and fitting
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting on the test dataset 
prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Number')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
st.subheader('Confusion Matrix and Accuracy Score')
st.write(cm)
accuracy_score(y_test, y_pred)
st.write(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
st.subheader('Classification Report')
st.write(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
st.subheader('K-Fold Validation')
st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
st.write("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
st.subheader('XGBoost Confusion Matrix and Accuracy Score')
st.write(cm)
accuracy_score(y_test, y_pred)
st.write(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
st.subheader('XGBoost Classification Report')
st.write(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
st.subheader('XGBoost K-Fold Validation')
st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
st.write("Standard Deviation: {:.2f} %".format(accuracies.std()*100))