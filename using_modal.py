import subprocess
import time

import cv2
import numpy as np
import pandas as pd

cascPath = './fer/haarcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)


def prepare_start():
    command = "mxnet-model-server --models fer=fer.mar --model-store ."
    return subprocess.check_output(command, shell=True)


def predic(file):
    command = 'curl -X POST http://127.0.0.1:8080/fer/predict -T ' + file
    data = subprocess.check_output(command, shell=True)
    str_res = str(data).replace("b'", "").replace("\\n", "")[:-1]
    data = pd.read_json(path_or_buf=str_res)
    # data.drop("neutral", axis=1, inplace=True)
    # data.drop(index=0, axis=0, inplace=True)
    col = data.columns.values
    best_val = 0, None
    for index, val in enumerate(np.array(data)):
        value = val[~np.isnan(val)]
        if value[-1] > best_val[0]:
            best_val = (value[-1], col[index])
    return best_val[-1]


def show_webcam():
    preTime = time.time()
    lable = ""
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
        )
        newTime = time.time()

        if len(faces) > 0:
            (x, y, w, h) = faces[-1]
            if newTime - preTime > 0.5:
                preTime = newTime
                frame_predict = cv2.resize(frame, (350, 350))
                cv2.imwrite("frame_predict.jpg", frame_predict)
                lable = predic("./frame_predict.jpg")
            cv2.putText(frame, lable,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0, 255, 0),
                        3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, lable,
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    3)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    subprocess.check_output("mxnet-model-server --stop", shell=True)


# prepare_start()
show_webcam()
