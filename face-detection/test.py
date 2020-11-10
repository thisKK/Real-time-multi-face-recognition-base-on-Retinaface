
import cv2
import numpy as np

from face_detection import RetinaFace

if __name__ == "__main__":
    detector = RetinaFace()
    cap = cv2.VideoCapture('../test.mp4')
    while True:
        ret, img = cap.read(0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(img)
        for box, landmarks, score in faces:
            box = box.astype(np.int)
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2
            )
        cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

