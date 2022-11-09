import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
import seaborn as sns

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header: 
	st.title('Iris Dataset with Streamlit!')
	st.text('This is a very popular datatset with machine learning models')

with dataset: 
	st.header('iris dataset')
	st.text('This dataset can be imported from the sklearn library. Lets look at at a few rows')

	iris = datasets.load_iris()
	irisData = pd.DataFrame(data=iris.data, columns=iris.feature_names)
	irisData['target'] = iris.target
	st.write(irisData.sample(5))
	st.header('Data visualizations')
	#fig, ax = plt.subplots(2,2)
	#fig.suptitle('Iris plots')
	#ax[0,0].scatter(x=irisData.iloc[:,0], y=irisData.iloc[:,1], c=irisData["target"])
	#ax[0,0].set_title('sepal length vs sepal width')
	#ax[0,1].scatter(x=irisData.iloc[:,0], y=irisData.iloc[:,2], c=irisData["target"])
	#ax[0,1].set_title('sepal length vs petal length')
	#ax[1,0].scatter(x=irisData.iloc[:,1], y=irisData.iloc[:,3], c=irisData["target"])
	#ax[1,0].set_title('sepal width vs pedal width')
	#ax[1,1].scatter(x=irisData.iloc[:,1], y=irisData.iloc[:,2], c=irisData["target"])
	#ax[1,1].set_title('sepal width vs petal length')
	#fig.set_figheight(8)
	#fig.set_figwidth(10)
	#st.pyplot(fig)
	fig = sns.pairplot(irisData, hue='target', palette = 'hls') 
	st.pyplot(fig)

with features: 
	st.header('Features in the dataset: ')

	#st.markdown(irisData.corr()['sepal length (cm)'])  #irisDatahome.corr()['sepeal length']
	iris_corr = irisData.corr()['target'][:-1]
	top_features = iris_corr[abs(iris_corr) > 0.5].sort_values(ascending=False) 
	st.markdown("There is {} strongly correlated values with the target value:\n{}".format(len(top_features), top_features))
	st.markdown('* **Sepal length:** has a covariance of .78')
	st.markdown('* **Petal length:** has a covariance of .94')
	st.markdown('* **Petal width:** has a covarnace of .95')


with modelTraining: 
	st.header('Training the model...')
	st.text('Choose the hyperparamters you want for the model')
	sel_col, disp_col = st.columns(2)
	models = ('KNN', 'RandomForest','SVC','Logisitc Regression')
	svc_params = ('rbf','linear','poly')
	model_selection = st.selectbox("Models:",models)

	if model_selection == models[0]:
		params = st.slider('How many neighbors do you want?',min_value=0, max_value=100,value=10, step=1)
	elif(model_selection == models[1]):
		params = st.slider('What is the max depth of the tree?',min_value=1, max_value=10,value=5, step=1)
	elif (model_selection == models[2]):
		params = st.selectbox("Kernel:",svc_params)
	else:
		st.text('We will use the default parameters for Logisitc Regression')
		params = ''

	training_button = st.button("Train Model",'')
	X, y = irisData.iloc[:,:-1], irisData.iloc[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=4)
	
	if (training_button):
		if(model_selection == models[0]):
			knn = KNeighborsClassifier(n_neighbors=params)
			knn.fit(X_train, y_train)
			y_pred = knn.predict(X_test)
			st.write("Accuracy:" ,metrics.accuracy_score(y_test, y_pred))
		elif model_selection == models[1]: 
			rf = RandomForestClassifier(n_estimators=10, max_depth=params, random_state=222)
			rf.fit(X_train, y_train)
			y_pred = rf.predict(X_test)
			st.write("Accuracy:" ,metrics.accuracy_score(y_test, y_pred))
		elif model_selection == models[2]:
			svc = SVC(kernel=params)
			svc.fit(X_train, y_train)
			y_pred = svc.predict(X_test)
			st.write("Accuracy:" ,metrics.accuracy_score(y_test, y_pred))
		else: 
			logreg = sk.linear_model.LogisticRegression(random_state=25)
			logreg.fit(X_train, y_train)
			y_pred = logreg.predict(X_test)
			st.write("Accuracy:" ,metrics.accuracy_score(y_test, y_pred))


	
