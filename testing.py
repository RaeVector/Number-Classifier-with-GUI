import tensorflow as tf
from keras.models import load_model
import cv2

model=load_model('basic_cnn.keras')
def cnn_model_test(image):
    image=cv2.imread('test.jpg',0)
    image=cv2.bitwise_not(image)
    image=cv2.resize(image,(28,28))
    image=image.reshape(1,28,28,1)
    image = image/255.0
    image=image.astype('float32')
    pred = model.predict(image)
    
    return pred