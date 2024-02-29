
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# CONV_2: add CONV layer with RELU activation and depth = 64 kernels
classifier.add(Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu')) # 30X30X32
classifier.add(Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu')) #28
classifier.add(Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu')) #26
#POOL_2: more downsampling
classifier.add(MaxPooling2D(pool_size=(2,2)))  # 13x13x64
# Adding a second convolutional layer
# CONV_2: add CONV layer with RELU activation and depth = 64 kernels
classifier.add(Conv2D(64, kernel_size=(3,3), padding = 'same', activation='relu')) # 11X11X64
classifier.add(Conv2D(64, kernel_size=(3,3), padding = 'same', activation='relu')) #9
classifier.add(Conv2D(64, kernel_size=(3,3), padding = 'same', activation='relu')) #7
#POOL_2: more downsampling
classifier.add(MaxPooling2D(pool_size=(2,2)))  # 4x4x32

# Adding a 3rd convolutional layer
# CONV_2: add CONV layer with RELU activation and depth = 64 kernels
classifier.add(Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu')) # 11X11X64
classifier.add(Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu')) #9
classifier.add(Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu')) #7
#POOL_2: more downsampling
classifier.add(MaxPooling2D(pool_size=(2,2)))  # 4x4x32

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 5, activation='softmax'))

# Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) for binary classes
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Flowers/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Flowers/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 1000)

classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('/Users/sudhanshukumar/Downloads/cat.11.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
#     print(prediction)
# else:
#     prediction = 'cat'
#     print(prediction)