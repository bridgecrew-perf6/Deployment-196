import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import plotly.express as px


st.title("Exploring Different Classifier")

st.write("""
#### Project by Immanuel
"""
)



dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Wine","Diabetes"))
#st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))
#st.write(classifier_name)

def get_dataset(datset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_diabetes()

    X=data.data
    Y=data.target
    return X,Y

X,Y = get_dataset(dataset_name)
st.write("Total Rows & Columns",X.shape)
st.write("Number of classes",len(np.unique(Y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K Value",1,10)
        params["K"]= K
        return params
    elif  clf_name == "SVM":
        C = st.sidebar.slider("C Value",0.02,10.0)
        params["C"]= C
        return params
    else:
        max_depth = st.sidebar.slider("Max_Depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        return params
    
params = add_parameter_ui(classifier_name)


def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
        
    elif  clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],random_state=123)
    
    return clf

clf = get_classifier(classifier_name, params)

# Classification
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=123)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)
acc = round((acc*100),2)

st.write(f"classifier = {classifier_name}")
st.subheader(f"Accuracy = {acc}%")


    
#Plotting

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]


#fig = plt.figure()
#plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
#plt.xlabel("Principal Component 1")
#plt.ylabel("Principal Component 2")
#plt.colorbar()

fig = px.scatter(x=x1,y=x2,color=Y)
st.plotly_chart(fig,use_container_width=False)
#st.pyplot(fig)



