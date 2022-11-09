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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

header = st.container()
dataset = st.container()
cleaning = st.container()
scale = st.container()
modelTraining = st.container()
test = st.container()

df = pd.read_csv('titanicTrain.csv')
data = df.iloc[:,[1,3,4]]
global X
global y
global X_train
global X_test
global y_train
global y_test

with header: 
	st.title('Titanic Dataset')
	st.subheader('We will use real data from the titanic and make preditions if you would have surviced the titanic!')


with dataset: 
	st.subheader('Lets load the dataset and take a look at it')
	load_button = st.button("Load Data",'')
	df = pd.read_csv('titanicTrain.csv')
	if load_button: 	
		st.write(df.sample(10))

with cleaning: 
	st.subheader('This is a lot of data to compact, lets clean it and only keep rows we need')
	clean_button = st.button('Clean Data',' ')
	data = df.iloc[:,[1,3,4]]
	if clean_button: 
		st.write('Completed')
		st.write(data.sample(10))
		st.subheader('Now that we have the data we want, we need to scale it, then remove any missing values')
	
with scale: 
	scale_button = st.button('Scale Data','  ')
	df_cleaned = data.dropna()
	X, y = df_cleaned.iloc[:,:], df.iloc[:,0]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
	ordinal_encoder = OrdinalEncoder(handle_unknown='ignore')
	ordinal_encoder.fit(X_train)
	X_train = ordinal_encoder.transform(X_train)
	ordinal_encoder = OrdinalEncoder()
	ordinal_encoder.fit(X_test)
	X_test = ordinal_encoder.transform(X_test)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	X_test = pd.DataFrame(X_test)
	X_train = pd.DataFrame(X_train)
	X_train.columns = {'Pclass','Sex','Age'}
	if scale_button: 
		st.write('Completed')
		st.write(X_train.head())
		st.subheader('This is a lot better, now lets train this model and make some predictions')

	

with modelTraining: 
	train_button = st.button("Train Data",'   ')
	clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=222)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	if train_button: 
		st.write('this model has an accuracy of: ',np.sum(y_test==pred)/len(pred))
	st.subheader('Type in your information and let the model guess what would have happend to you!')

with test: 
	pclass = st.slider('What passenger class were you on?',min_value=1, max_value=3,value=2, step=1)
	age = st.text_input('Enter your age')
	sexs = {'Male', 'Female'}
	sex = st.selectbox('Sex', sexs)
	if sex == 'Male':
		sex=0
	else:
		sex=1
	person = pd.DataFrame([[sex, pclass, age]])
	predict_button = st.button("Predict Data",'    ')
	if predict_button: 
		pred_person = clf.predict(person)
		if(pred_person < 50):
			st.write('You didnt make it :(')
		else: 
			st.write('Youre Alive!!!')
		





