import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import datetime
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

from scipy import misc
import cv2

# Function to crop, blur, convert color space, and resize an image
# Input array must be 160x320 (rows by columns) BGR formatted image
# Output array is 18x80
def cropResize(image):
    imgNew= image[62:134,:]
    imgNew= cv2.bilateralFilter(imgNew,7,0,75)
    imagePlex= (cv2.cvtColor(imgNew, cv2.COLOR_BGR2HSV))[:,:,1]
    return misc.imresize(imagePlex, (18,80,1))

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
qS1,qS2,qS3 = [], [], []
start_time= datetime.datetime.now()
def findNearest(arr, val):
    i= (np.abs(arr-val)).argmin()
    return arr[i], i

@sio.on('telemetry')
def telemetry(sid, data):
    global start_time, qS1, qS2, qS3
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array= cropResize(np.asarray(image))
    transformed_image_array= np.reshape(image_array, (-1, 18,80,1))
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # average the model's steering value for the past 3,9,18 frames...
    qS1.insert(0,steering_angle)
    qS2.insert(0,steering_angle)
    qS3.insert(0,steering_angle)
    if len(qS1)>3: qS1.pop()
    if len(qS2)>9: qS2.pop()
    if len(qS3)>18: qS3.pop()
    # place averages into an array
    avg= np.array([sum(qS1)/float(len(qS1)), sum(qS2)/float(len(qS2)), sum(qS3)/float(len(qS3)), 0])
    # find the nearest steering angle
    steering_angle, idx= findNearest(avg, steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # back off the throttle when steering angle is more than 0.1 (2.5 degrees)
    if (speed>18.0) and (abs(steering_angle)>0.1):
        throttle= 0.0
    else:
        if speed<10.0: # apply full throttle when speed is low
            throttle= 1.0
        else:
            throttle= 0.2
    delta= datetime.datetime.now() - start_time
    print("dir: {0:+.6f}, pwr: {1:.3f}, vel: {2:.2f}, fps: {3:.2f}, idx: {4:d}".format(steering_angle,throttle, speed, 1e6/delta.microseconds, idx))
    start_time= datetime.datetime.now()
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        #model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    model.summary()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)