import cv2
import numpy as np
import os
import time
import pickle

from face_detection import RetinaFace
path = '../data/29--Students_Schoolkids/'
# model = 'resnet50'
model = 'mobilenet0.25'
scale = '1'
name = 'retinaFace'
count = 0
CONFIDENCE = 0.1
if __name__ == "__main__":
    for fn in os.listdir(path):
        filename = fn
        raw_img = cv2.imread(os.path.join(path, filename))
        detector = RetinaFace()
        out_file = '../data'
        name = fn.split('.')
        name = name[0]
        out_file = os.path.join(out_file, name.replace('jpg', 'txt'))
        t0 = time.time()
        print('start')
        faces = detector(raw_img)
        t1 = time.time()
        print(f'took {round(t1 - t0, 3)} to get {len(faces)} faces')
        # with open(out_file + '.txt', 'w') as f:
        #     # f.write("%s\n" % str(name))
        #     # f.write("%d\n" % len(faces))
        for box, landmarks, score in faces:
            box = box.astype(np.int)
            if score > CONFIDENCE:
                with open(out_file + '.txt', 'a') as f:
                    f.write("%s %g %d %d %d %d\n" % (str('face'), score, box[0], box[1], box[2] - box[0], box[3] - box[1]))
                    # f.write("%d %d %d %d %g\n" % (box[0], box[1], box[2] - box[0], box[3] - box[1], score))
                # cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)

        # while True:
        #     cv2.imshow('IMG', raw_img)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

