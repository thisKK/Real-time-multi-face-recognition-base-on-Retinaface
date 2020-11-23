# Real-time-multi-face-recognition-base-on-Retinaface
Requirements

    Python 3.5+ (it may work with other versions too)
    Linux, Windows or macOS
    PyTorch (>=1.0)
    opencv

#FaceDetecter
Install
1. git clone https://github.com/thisKK/Real-time-multi-face-recognition-base-on-Retinaface.git
3. install Requirements
    - The easiest way to install it is using pip:
      pip install face-detection
      or
      pip install git+https://github.com/elliottzheng/face-detection.git@master
2. dowload and put weight in to weigth folder 

#FaceRecognition
Requirements
    dlib
    faceDetection #our use Retinaface
    
How to use 
1.put video only one person to train model in to folder 
    ..faceRecognition/Video/
      --name1
      --name2
2.run Extrackvideo.py if sucess you will get trainset.pk file 
3.run FaceRecognition2.py for detect face and reconition. Any way you can edit video path 
