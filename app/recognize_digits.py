from keras import models, backend
import numpy as np
import pickle
import base64
from scipy.misc import imread, imresize


def parse_image(uri):
    imgstr = uri.split(',')[1]

    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


def recognize(uri):
    # Read image
    parse_image(uri)
    img = imread('output.png', mode='L')
    img = np.invert(img)
    img = imresize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)

    # Clear session to allow repeated predictions
    backend.clear_session()

    # Load model and Label Binarizer
    try:
        model
    except (NameError, UnboundLocalError):
        model = models.load_model('app/model/dr.h5')
        with open('app/model/label_binarizer.pkl', 'rb') as f:
            lb = pickle.load(f)

    # Make prediction
    pred = model.predict(img)
    ind = pred.argmax(axis=1)[0]
    label = lb.classes_[ind]

    return str(label), str(pred[0][ind]*100)
