import cv2
import numpy as np
import matplotlib.pyplot
from PIL import ImageFont, ImageDraw, Image
from time import time
from tensorflow.keras.models import load_model
from database import insert_fer
import face_recognition
matplotlib.use('TkAgg')

class_name = ["bình thường", "vui", "ngạc nghiên", "buồn", "tức giận", "kinh tởm", "sợ hãi", "khinh thường"]

my_model = load_model("model.h5")

face_size = 48

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

saved_faces = []
saved_emotions = []


def run_pre(image_org):
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    height, width, _ = image_org.shape
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    crop_faces_gray = []
    crop_faces_color = []
    crop_size = []

    for face in faces:
        (x, y, w, h) = face
        image = image_org[y:y + h, x:x + w]
        cv2.rectangle(image_org, (x, y), (x + w, y + h), (255, 0, 0), 3)
        image = cv2.resize(image, dsize=(face_size, face_size))
        crop_faces_color.append(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image).reshape(face_size, face_size, 1)
        crop_faces_gray.append(image)
        crop_size.append((x, y, w, h))
    if len(crop_faces_gray) > 0:
        predicts = my_model.predict(np.array(crop_faces_gray))
        for idx, predict in enumerate(predicts):
            if np.max(predict) >= 0.3:
                id_name = np.argmax(predict)
                name = class_name[id_name]

                face_encoding = face_recognition.face_encodings(crop_faces_color[idx])
                if face_encoding.__len__() > 0:
                    face_encoding = face_encoding[0]
                    compare_faces = face_recognition.compare_faces(saved_faces,face_encoding)
                    if True in compare_faces:
                        idx_face_already_have = compare_faces.index(True)
                        if id_name != 0 and saved_emotions[idx_face_already_have]!=id_name:
                            saved_emotions[idx_face_already_have] = id_name
                            insert_fer(name,int(id_name))
                        
                    if True not in compare_faces:
                        saved_faces.append(face_encoding)
                        saved_emotions.append(id_name)
                        if id_name != 0:
                            insert_fer(name,int(id_name))
                x, y, w, h = crop_size[idx]
                org = (x, y)
                color = (0, 255, 0, 0)
                font_size = int((w / width + h / height) * 50)
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
    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break
