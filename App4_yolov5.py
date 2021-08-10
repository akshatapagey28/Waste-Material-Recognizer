# import the necessary packages
import torch
import numpy as np
import argparse
import time
import cv2
import os
from flask import Flask, request, Response, jsonify, render_template
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image
import base64

#--------------
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras
from PIL import Image


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom


# def get_class():
#     img = Image.open('objectImage.jpg')
#     img = img.resize((384,512)) #as we use original dimention for training model
#     # display(img)
#     # x = image.img_to_array(img)
#     x = keras.preprocessing.image.img_to_array(img)
#     x = x/255
#     x = np.expand_dims(x, axis=0)

#     images = np.vstack([x])
#     classes = model_classifier.predict(images)
#     object_class = labels[np.argmax(classes)]
#     print('prediction : ', object_class)   
#     return object_class 


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


# def get_predection(image):


#     # construct a blob from the input image and then perform a forward
#     # pass of the YOLO object detector, giving us our bounding boxes and
#     # associated probabilities
#     bbox, label, conf = cv.detect_common_objects(image, model='yolov5s')
#     print('-----Detection is done-------')
#     lableIndex=0
#     frame = draw_bbox(image, bbox, label, conf)
#     for b in bbox:    
        
#         cv2.imwrite('output.jpg',frame)
#         # startY:endY, startX:endX
#         x,y,w,h=b
#         # print(b)
#         objectImage=image[y:h , x:w]
		
# 		# objectImage=img.crop((x,y,x+w,y+h))
#         cv2.imwrite('objectImage.jpg',objectImage)
#         objectType=get_class()
#         # pred=model_predict(objectImage, model)
#         print('----label--------',label[lableIndex])
#         lableIndex=lableIndex+1
#         print('pred of object', objectType)
#         frame = cv2.putText(frame, objectType, (x+int(w/2),y-10), cv2.FONT_HERSHEY_SIMPLEX,
#                    1, (255, 0, 0), 2, cv2.LINE_AA)
#     print('returning')
#     return frame
   

# Initialize the Flask application
app = Flask(__name__)
# run_with_ngrok(app)  

@app.route('/')
def home():
    return render_template('index.html')


# route http posts to this method
@app.route('/', methods=['POST'])
def main():
    # load our input image and grab its spatial dimensions
    #image = cv2.imread("/content/rsz_namibia_will_burrard_lucas_wwf_us_1.jpg")
    
    img = request.files["file"].read()
    # print('-----imagename-------')
    # print(img)
    # print('-----END imagename-------')
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    results = model(image)
    labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()


    for label in labels:
        print('label ',label)
       

    # res=get_predection(image)
    results.save()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    #cv2_imshow(res)
    #cv2.waitKey()
    #cv2.imwrite("filename.png", res)
    np_img=Image.fromarray(image)
    img_encoded=image_to_byte_array(np_img)  
    base64_bytes = base64.b64encode(img_encoded).decode("utf-8")    
    #return jsonify({'status': True, 'image': image})      
    return render_template('index.html', user_image=base64_bytes)

    # start flask app
if __name__ == '__main__':
    app.run()