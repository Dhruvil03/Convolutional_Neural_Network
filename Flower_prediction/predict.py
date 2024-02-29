import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class flower:
    def __init__(self,filename):
        self.filename =filename


    def predictionflower(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(self.filename, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        if np.argmax(result) == 1:
            prediction = 'dandelion'
            return [{ "image" : prediction}]
        elif np.argmax(result) == 2:
            prediction = 'rose'
            return [{ "image" : prediction}]
        elif np.argmax(result) == 3:
            prediction = 'sunflower'
            return [{ "image" : prediction}]
        elif np.argmax(result) == 4:
            prediction = 'tulip'
            return [{ "image" : prediction}]
        elif np.argmax(result) == 0:
            prediction = 'daisy'
            return [{ "image" : prediction}]
        else:
            prediction = 'Error'
            return [{"image": prediction}]

# def predictionflower(self):
#     # Load model
#     model = load_model('model.h5')
#
#     # Load and preprocess the image
#     test_image = image.load_img(self.filename, target_size=(64, 64))
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis=0)
#
#     # Make prediction
#     result = model.predict(test_image)
#
#     # Map the prediction to flower class
#     flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
#     predicted_class_index = np.argmax(result)
#     prediction = flower_classes[predicted_class_index]
#
#     return [{"image": prediction}]
