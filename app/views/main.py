from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


@app.route('/')

#disease_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', \
                  # 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', \
                  # 'Hernia']

@app.route('/upload')
def upload_file2():
   return render_template('index.html')
	
@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      #train_positive=os.listdir(path)
      #df = pd.read_csv("pneumonia",engine="python")
      #preprocess_input, test_data = fm(df)
      # l=[]
      # for file_ in train_positive:
      #    if file_[-5:]=='.jpeg':
      #       l.append(file_)
      
      myArr = np.zeros((1,150,150,3))
      # for x,img in enumerate(l):
      #    im = cv2.imread(path+str(img))
      #    myImg = cv2.resize(im,(150,150))
      #    myImg = myImg/255.
      #    myArr[x,:,:,:] = myImg

      im = cv2.imread(path)
      myImg = cv2.resize(im,(150,150))
      myImg = myImg/255.
      myArr[0,:,:,:] = myImg


      model=mymodel()
      model.load_weights('model_weights.h5') #pretrained model
    
      PREDS = model.predict(myArr)
      
      f.save(path)
   return render_template('uploaded.html', title='Success', predictions=PREDS, user_image=f.filename)


def mymodel():
   image_height = 150
   image_width = 150
   batch_size = 32
   no_of_epochs  = 10
   
   model = Sequential()
   model.add(Conv2D(32,(3,3),input_shape=(image_height,image_width,3),activation='relu'))
   model.add(Conv2D(32,(3,3),activation='relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Dropout(0.2))
   model.add(Conv2D(64,(3,3),activation='relu'))
   model.add(Conv2D(64,(3,3),activation='relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Dropout(0.2))
   model.add(Conv2D(128,(3,3),activation='relu'))
   model.add(Conv2D(128,(3,3),activation='relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Dropout(0.2))
   model.add(Flatten())
   model.add(Dense(units=128,activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(units=1,activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model


@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')