import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps


X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=2500,train_size=7500,random_state=9)
X_train=X_train/255
X_test=X_test/255
classifier=LogisticRegression(solver="saga",multi_class="multinomial")
classifier.fit(X_train,y_train)

def getPrediction(image):
    im_PIL=Image.open(image)
    im_BW=im_PIL.convert("L")
    im_BW_resize=im_BW.resize((28,28),Image.ANTIALIAS)
    pixel_filter=20
    min_pixel=np.percentile(im_BW_resize,pixel_filter)
    im_BW_resize_inverted_scale=np.clip(im_BW_resize-min_pixel,0,255)
    max_pixel=np.max(im_BW_resize)
    im_BW_resize_inverted_scale=np.asarray(im_BW_resize_inverted_scale)/max_pixel
    test_sample=np.array(im_BW_resize_inverted_scale).reshape(1,284)
    test_predict=classifier.predict(test_sample)
    return test_predict[0]