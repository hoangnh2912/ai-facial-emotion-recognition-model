
import cv2
import numpy as np
import matplotlib.pyplot
from PIL import ImageFont, ImageDraw, Image
from time import time

matplotlib.use('TkAgg')

from tensorflow.keras.models import load_model

class_name = ["bình thường", "vui", "ngạc nghiên", "buồn", "tức dận", "tởm", "sợ", "contempt"]

my_model = load_model("model.h5")


def run_pre(image_org):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    height, width, _ = image_org.shape
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    crop_faces = []
    crop_size = []
    for face in faces:
        (x, y, w, h) = face
        image = image_org[y:y + h, x:x + w]
        cv2.rectangle(image_org, (x, y), (x + w, y + h), (255, 0, 0), 3)
        image = cv2.resize(image, dsize=(48, 48))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image).reshape(48, 48, 1)
        crop_faces.append(image)
        crop_size.append((x, y, w, h))
        # predict = my_model.predict()
        # name = class_name[np.argmax(predict)]
    if len(crop_faces) > 0:
        predicts = my_model.predict(np.array(crop_faces))
        for idx, predict in enumerate(predicts):
            if np.max(predict) >= 0.3:
                name = class_name[np.argmax(predict)]
                x, y, w, h = crop_size[idx]
                org = (x, y)
                color = (0, 255, 0, 0)
                font_size = int((w / width + h / height) * 100)
                img_pil = Image.fromarray(image_org)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype('OpenSans-Regular.ttf', font_size)
                draw.text(org, name, fill=color, font=font)
                image_org = np.array(img_pil)
    return image_org


cap = cv2.VideoCapture(0)
start = time()
font = cv2.FONT_HERSHEY_SIMPLEX
while (1):
    ret, frame = cap.read()
    processed = run_pre(frame)
    now = time()
    cv2.putText(processed, str(round(1 / (now - start), 1)) + "FPS", (50, 50), fontFace=font
                , fontScale=1, color=(0, 255, 0), thickness=2)
    start = now
    cv2.imshow("cam", processed)
    # cv2.imshow("cam", img)
    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break
