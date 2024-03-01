import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#Page Setup
st.set_page_config(
    page_title = 'Iris Flower Prediction App',
    layout = 'centered',
    initial_sidebar_state = 'expanded'
)

st.title("""
Iris Flower Prediction App
""")

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

st.sidebar.layout = 'centered'
st.subheader('User Input Parameters')
#st.write(df)
st.dataframe(df, hide_index = True)
#st.sidebar.subheader('User Input Parameters')
#st.sidebar.write(df)
st.sidebar.dataframe(df, hide_index = True)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

st.subheader('Iris Data Table')
#st.write(pd.iris)
df_iris = pd.DataFrame(X, columns = iris.feature_names)
df_iris['target'] = iris['target']
#df_iris.head()
#st.dataframe(df_iris, hide_index = True)

df_iris1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target']).astype({'target': int}) \
       .assign(species=lambda x: x['target'].map(dict(enumerate(iris['target_names']))))
st.dataframe(df_iris1, hide_index = True)

st.subheader('EDA - Mean Grouped by Class')
df_iris2 = df_iris1.drop(['target'], axis = 1 )
groupby_species_mean = df_iris2.groupby('species').mean()
st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)

st.subheader('Class labels and Their Corresponding Index #')
df1 = pd.DataFrame(df_iris1.target.dropna().unique(), columns=['Species #'])
df2 = pd.DataFrame(df_iris1.species.dropna().unique(), columns=['Species'])
st.dataframe(pd.concat([df1, df2], axis = 1, ), hide_index = True)

#Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

#Scaling but this creates a lot of flux in MAE
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the random forest classifier model and fitting
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth = 2, max_features = 4, n_estimators = 200, random_state = 42)
classifier.fit(X_train, y_train)

# Predicting on the test dataset 
prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.subheader('Prediction & Prediction Number')
p1 = pd.DataFrame(iris.target_names[prediction], columns = ['Species'])
p2 = pd.DataFrame(prediction, columns = ['Species #'])
st.dataframe(pd.concat([p1, p2], axis = 1), hide_index = True)
#st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
#st.write(prediction_proba)
st.dataframe(prediction_proba, hide_index = True)
#pp = pd.DataFrame(prediction_proba, columns = ['Prob 1', 'Prob 2'])
#st.dataframe(pp, hide_index = True)

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