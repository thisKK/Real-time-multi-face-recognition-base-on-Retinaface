import cv2
import numpy as np
from face_detection import RetinaFace
model = 'resnet50'
# model = 'mobilenet0.25'
cap = cv2.VideoCapture('../TestVideo/test.mp4')

if __name__ == "__main__":
    while True:
        detector = RetinaFace()
        _, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(frame)
        print(faces)
        for box, landmarks, score in faces:
            box = box.astype(np.int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

