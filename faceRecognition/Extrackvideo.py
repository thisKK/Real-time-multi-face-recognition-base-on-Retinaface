import cv2
import os
import numpy as np
import dlib
import pickle
from faceDetection import RetinaFace

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../facialLandmarks/shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('../faceRecognition/dlib_face_recognition_resnet_model_v1.dat')

path_output = './facedata/'
TRAINING_BASE = '../faceRecognition/Video'

dirs = os.listdir(TRAINING_BASE)
images = []
names = []

FACE_DESC = []
FACE_NAME = []

if __name__ == "__main__":
    face_detector = RetinaFace(gpu_id=0)
    print("load retina face done!!")
    
    for label in dirs:
        for i, fn in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))):
            print(f"start collecting faces from {label}'s data")
            cap = cv2.VideoCapture(os.path.join(TRAINING_BASE, label, fn))
            frame_count = 0
            while True:
                # read video frame
                ret, raw_img = cap.read()
                # process every 5 frames
                if frame_count % 5 == 0 and raw_img is not None:
                    h, w, _ = raw_img.shape
                    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    faces = face_detector(img)
                    for box, landmarks, score in faces:
                        # if face detected
                        if box.shape[0] > 0:
                            score.astype(np.int)
                            box = box.astype(np.int)
                            if score < 0.4:
                                continue
                            # cropped = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]][:, :, :, : -1]
                            # cropped = img[box[1]:box[3], box[0]:box[2]][:, :, ::-1] #crop face
                            # cropped = cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_AREA) # resize face
                            dRect = dlib.rectangle(left=box[0], top=box[1],
                                                   right=box[2],
                                                   bottom=box[3])  # transform Opencv rectangle to dlib rectangle format
                            shape = sp(raw_img, dRect)
                            face_desc = model.compute_face_descriptor(raw_img, shape, 200)
                            FACE_DESC.append(face_desc)
                            FACE_NAME.append(label)
                            # cv2.imwrite(f'faces/tmp/{label}_{frame_count}.jpg', cropped)
                frame_count += 1
                if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print('ADD FACE DONE')

                    break
    pickle.dump((FACE_DESC, FACE_NAME), open('trainset.pk', 'wb'))
    print('successful')