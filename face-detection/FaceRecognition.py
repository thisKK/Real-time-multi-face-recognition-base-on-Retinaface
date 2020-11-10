import cv2
import numpy as np
from face_detection import RetinaFace
import dlib
import time
import pickle

# face_detector = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')

# ---------- load face landmark predictor  --------------------------
sp = dlib.shape_predictor('../FacialLandmarks/shape_predictor_68_face_landmarks.dat')

# ---------- load dlib detector  --------------------------
detector = dlib.get_frontal_face_detector()
# ---------- load resnet model for recognition --------------------------
model = dlib.face_recognition_model_v1('../facerecognition/dlib_face_recognition_resnet_model_v1.dat')

# ---------- load face bank  --------------------------
FACE_DESC, FACE_NAME = pickle.load(open('../facerecognition/trainset.pk', 'rb'))

# ---------- read video --------------------------
# cap = cv2.VideoCapture('0') #read from web camera
cap = cv2.VideoCapture('../TestVideo/classroom.mp4')

# ---------- write out video result -----------------
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1920, 1080))
cap.set(3, 1920)
cap.set(4, 1080)

if __name__ == "__main__":
    # ---------- call class retina face detector --------------------------
    face_detector = RetinaFace()
    print("load retina face done!!")

    while True:
        t0 = time.time()
        isSuccess, frame = cap.read()
        if isSuccess:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_detector(frame)
                for box, landmarks, score in faces:
                    score.astype(np.int)
                    if score < 0.3:
                        continue
                    box = box.astype(np.int)
                    face = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][:, :, ::-1]  # face position
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
                    dets = detector(face, 1)  # transform Opencv rectangle to dlib rectangle format
                    for k, d in enumerate(dets):
                        shape = sp(face, d)
                        # check distance between facebank and prediction
                        face_desc0 = model.compute_face_descriptor(face, shape, 1)
                        distance = []
                        for face_desc in FACE_DESC:
                            distance.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
                        distance = np.array(distance)
                        idx = np.argmin(distance)
                        if distance[idx] < 0.4:
                            name = FACE_NAME[idx]
                            cv2.putText(frame, name, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                        (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, 'unknow', (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                        (255, 255, 255), 2)
            except:
                pass

            cv2.imshow("", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        t1 = time.time()
        print("frame")
        print(f'took {round(t1 - t0, 3)} to process')

    print("Suecess")




