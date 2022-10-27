import streamlit as st
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf

header = st.container()
dataset = st.container()
train = st.container()
model_cnn = st.container()
test = st.container()

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes = 10
y_train = keras.utils.to_categorical(y_train, classes) 
y_test = keras.utils.to_categorical(y_test, classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

col1, col2, col3 = st.columns([1,2,3])

with header: 
	st.title('Image recognition with Streamlit!')
	st.text('We will create a CNN to recognize images from the Cifar10 dataset,\nthen upload images and classify them')
with dataset: 
	st.title('The Cifar10 dataset has ten total target varibales in the dataset, they are: ')
	st.markdown('* **airplane** \u2708')
	st.markdown('* **car** \uF699')
	st.markdown('* **bird**')
	st.markdown('* **cat**')
	st.markdown('* **deer**')
	st.markdown('* **dog**')
	st.markdown('* **frog**')
	st.markdown('* **horse**')
	st.markdown('* **ship**')
	st.markdown('* **truck**')
	st.markdown('We will train a CNN model on this dataset!')
	
with train:
	st.title('Cifar10 Dataset')
	load_button = st.button("Load Dataset",'')
	if load_button:
		st.write('Data upload successful!')
	st.write('Click to see a random image in the dataset')
	view_button = st.button("View random image",' ')
	if view_button: 
		ran = random.randint(0,1000)
		fig = (X_train[ran])
		st.image(fig, width=500)
with model_cnn: 
	st.title('Creating a CNN')
	params = st.slider('How many hidden layers would you like?',min_value=1, max_value=5,value=2, step=1)
	params2 = st.slider('How many epochs?',min_value=1, max_value=10,value=5, step=1)
	model = Sequential()
	model.add(Conv2D(32, (5, 5), strides = (2,2), padding='same',input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	cre_button = st.button("Create CNN",'  ')
	if cre_button: 
		#for i in range(params):
		model.add(Conv2D(32, (5, 5), strides = (2,2)))
		model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten()) 
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation('softmax'))
		batch_size = 32
		opt = keras.optimizers.RMSprop(learning_rate=0.0005, decay=1e-6)
		model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		model.fit(X_train, y_train,batch_size=batch_size,epochs=params2,validation_data=(X_test, y_test),shuffle=True)
		predict_x=model.predict(X_test) 
		classes_x=np.argmax(predict_x,axis=1)
		np.argmax(y_test, axis=1)
		from sklearn.metrics import accuracy_score
		st.write('The accuracy of this model is: ',accuracy_score(np.argmax(y_test, axis=1), classes_x))	
		
with test: 
	st.header('upload a photo and test our model')
	uploaded_photo = col2.file_uploader('upload a photo')
	st.write('now lets make a prediction')
	pred_button = st.button("Predict Image",'   ')
	if pred_button: 
		image = tf.keras.preprocessing.image.load_img(uploaded_photo, target_size = (32, 32))
		input_arr = tf.keras.preprocessing.image.img_to_array(image)
		input_arr = np.array([input_arr])  # Convert single image to a batch.
		predictions = model.predict(input_arr/255.0)
		st.write(predictions)

