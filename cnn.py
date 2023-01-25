import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.callbacks import *
from keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, Input, MaxPooling2D
# create a new generator
imagegen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, brightness_range=[0.2,1.2])

# load train data
train = imagegen.flow_from_directory("../input/pothole-detection-dataset/Dataset/train", class_mode="categorical", shuffle=False, batch_size=128)

# load val data
val = imagegen.flow_from_directory("../input/pothole-detection-dataset/Dataset/val", class_mode="categorical", shuffle=False, batch_size=128)
test_path = '../input/pothole-detection-dataset/Dataset/test'

testing = []
for i in os.listdir(test_path):
  if i[:3] == 'nor':
    testing.append('Normal')
  else:
    testing.append('Pothole')
    input_img = Input(shape = (256, 256, 3))

tower_1 = Conv2D(25, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(25, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(25, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(25, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(25, (1,1), padding='same', activation='relu')(tower_3)
output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
output = Flatten()(output)
out    = Dense(2, activation='softmax')(output)
model1 = Model(inputs = input_img, outputs = out)

# finding the best model
mc1= ModelCheckpoint('best_model_inception.h5', monitor='val_accuracy', mode='max', save_best_only=True,verbose=1)

# compile model
model1.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fit on data for 50 epochs
history1 = model1.fit(train, epochs=50, validation_data=val, callbacks = [mc1])

#lading the best model
model1 = load_model('best_model_inception.h5')

# model summary
model1.summary()

(eval_loss1, eval_accuracy1) = model1.evaluate(val, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy1 * 100)) 
print("[INFO] Loss: {}".format(eval_loss1))
prediction1 = []

for i in os.listdir(test_path):
  img = image.load_img(test_path+'//'+i, target_size = (256,256))
  #plt.imshow(img)
  #plt.show()

  X = image.img_to_array(img)
  X = np.expand_dims(X, axis = 0)
  images = np.vstack([X])

  pred1 = model1.predict(images)

  if (pred1[0][0] > 0.6):
    prediction1.append('Normal')
  else:
    prediction1.append('Pothole')
    c = 0
for i in range(len(prediction1)):
  if prediction1[i] == testing[i]:
    c = c+1
accuracy1 = (c/(len(prediction1)))*100
