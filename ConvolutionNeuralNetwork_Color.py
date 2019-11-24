#from tensorflow import keras
import tensorflow as tf
#import numpy as np
#import keras_preprocessing
#from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

train_data_path = 'FinalColor/Training'
test_data_path = 'FinalColor/Testing'

TrainDataGen = ImageDataGenerator(rescale = 1./255)
TestDataGen = ImageDataGenerator(rescale = 1./255)

TrainSet = TrainDataGen.flow_from_directory(train_data_path,target_size=(100,100),class_mode='categorical')
TestSet = TestDataGen.flow_from_directory(test_data_path,target_size=(100,100),class_mode='categorical')

cnn_model =  tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(14, activation='softmax')
])

"""cnn_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))"""""


cnn_model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = cnn_model.fit_generator(TrainSet,epochs=25,validation_data = TestSet,verbose = 1)

