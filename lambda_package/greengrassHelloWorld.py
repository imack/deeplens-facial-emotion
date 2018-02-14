#
# Copyright Amazon AWS DeepLens, 2017
#

import os
import greengrasssdk
from threading import Timer
import time
import awscam
import cv2
import numpy as np
import mo
from threading import Thread

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

ret, frame = awscam.getLastFrame()
ret, jpeg = cv2.imencode('.jpg', frame)
Write_To_FIFO = True


class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path, 'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue


def greengrass_infinite_infer_run():
    try:
        model_type = "ssd"
        input_width = 224
        input_height = 224
        prob_thresh = 0.25
        results_thread = FIFO_Thread()
        results_thread.start()

        haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Face detection starts now")

        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}

        error, modelPath = mo.optimize('smile-net', input_width, input_height)

        model = awscam.Model(modelPath, mcfg)
        client.publish(topic=iotTopic, payload="Model loaded")
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")

        doInfer = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")

            img_copy = np.copy(frame)
            # convert the test image to gray image as opencv face detector expects gray images
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

            # let's detect multiscale (some images may be closer to camera than others) images
            faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);

            # go over list of faces and draw them as rectangles on original colored img
            for face in faces:
                (x, y, w, h) = face

                face_img = img_copy[y:y + h, x:x + w]
                resize_face = cv2.resize(face_img, (224, 224))

                infer_output = model.doInference(resize_face)
                parsed_results = model.parseResult(model_type, infer_output)
                prob = parsed_results[model_type][0]['prob']

                if prob > 0.0:
                    # smiling, show blueish
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 20), 4)
                else:
                    # Neutral, show orangish
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 165, 255), 4)

            msg = "{"
            msg += '"{}":{:.2f},'.format('smile', prob)
            msg += "}"
            client.publish(topic=iotTopic, payload=msg)
            global jpeg
            ret, jpeg = cv2.imencode('.jpg', frame)

    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return