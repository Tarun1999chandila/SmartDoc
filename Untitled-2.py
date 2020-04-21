#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'AI_Startup_Prototype/flaskSaaS-master'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

get_ipython().run_line_magic('matplotlib', 'inline')


#%%
total_images_train_normal = os.listdir(r"C:\Users\User\Downloads\chest_xray\train\NORMAL")
total_images_train_pneumonia = os.listdir(r"C:\Users\User\Downloads\chest_xray\train\PNEUMONIA")


#%%
s


#%%
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


#%%
model.summary()


#%%
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   shear_range=0.2,
                                   zoom_range=0.2
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)


#%%
training_set = train_datagen.flow_from_directory("C:\\Users\\User\\Downloads\\chest_xray\\train\\",
                                                 target_size=(image_width, image_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(r"C:\\Users\\User\\Downloads\\chest_xray\\test",
                                            target_size=(image_width, image_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

# Updated part --->
val_set = test_datagen.flow_from_directory(r"C:\\Users\\User\\Downloads\\chest_xray\\val",
                                            target_size=(image_width, image_height),
                                            batch_size=1,
                                            shuffle=False,
                                            class_mode='binary')


#%%
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]


#%%
# history = model.fit_generator(training_set,
#                     steps_per_epoch=5216//batch_size,
#                     epochs=no_of_epochs,
#                     validation_data=test_set,
#                     validation_steps=624//batch_size,
#                     callbacks=callbacks
#                    )

#run if u want to train


#%%
#model.save_weights('model_weights.h5')


#%%
model.load_weights('model_weights.h5') #pretrained model


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%

print(test_set.class_indices)


#%%
preds = model.predict_generator(test_set, steps = 32)


#%%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


#%%
predsBin = []


#%%
for x in preds:
    if(x>=0.5):
        predsBin.append(1)
    else:
        predsBin.append(0)


#%%
len(predsBin)


#%%
model.evaluate_generator(test_set, steps = 32)


#%%
#loss is 0.29 accuracy is 0.88


#%%



#%%



#%%



#%%



#%%
train_positive=os.listdir("C:\\Users\\User\\Downloads\\chest_xray\\test\\PNEUMONIA\\")
#df = pd.read_csv("pneumonia",engine="python")
#preprocess_input, test_data = fm(df)
l=[]
for file in train_positive:
    if file[-5:]=='.jpeg':
        l.append(file)


#%%
myArr = np.zeros((len(l),150,150,3))
for x,img in enumerate(l):
    im = cv2.imread("C:\\Users\\User\\Downloads\\chest_xray\\test\\PNEUMONIA\\"+str(img))
    myImg = cv2.resize(im,(150,150))
    myImg = myImg/255.
    myArr[x,:,:,:] = myImg
    


#%%
PREDS = model.predict(myArr)


#%%
PREDS


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%
predsFinal = []
for X in PREDS:
    if(X>=0.5):
        predsFinal.append(1)
    else:
        predsFinal.append(0)
    


#%%
len(l)


#%%
np.array(predsFinal).sum()


#%%
389/390


