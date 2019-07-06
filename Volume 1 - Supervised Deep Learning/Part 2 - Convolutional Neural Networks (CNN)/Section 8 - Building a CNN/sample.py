#CNN

#importing keras lib and packages
from keras.models import Sequential
from keras.layers import Conv2D ,MaxPooling2D ,Flatten ,Dense

#building cnn

#initialising
classifier = Sequential()

#1.convolution
classifier.add(Conv2D(32 ,(3 ,3) ,activation= 'relu' ,input_shape = (64 ,64 ,3)))

#2.max pooling
classifier.add(MaxPooling2D(pool_size = (2 ,2)))

#3.flattening
classifier.add(Flatten())

#4. full connection
classifier.add(Dense(units= 128 ,activation= 'relu'))
classifier.add(Dense(units= 1 ,activation= 'sigmoid'))

#compiling cnn
classifier.compile(optimizer= 'adam' ,loss= 'binary_crossentropy' ,metrics= ['accuracy'])

#fitting cnn
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=2000)

training_set.class_indices
# for single prediction
import numpy as np
from keras.preprocessing import image
test_img = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size= (64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis= 0)
result = classifier.predict(test_img)
if result[0][0] == 1 :
    print('dog')
else:
    print('cat')
