import numpy as np
import cv2
import sys
import os

class TrainLBPHFace:
    def __init__(self):
        cascPath = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        if not os.path.isdir('trained_data'):
            os.mkdir('trained_data')
        self.model = cv2.face.LBPHFaceRecognizer_create()

    def lbph_train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]

        self.model.train(imgs, tags)
        self.model.save('trained_data/lbph_trained_data.xml')
        print "Training completed successfully"
        return


if __name__ == '__main__':
    trainer = TrainLBPHFace()
    trainer.lbph_train_data()
    print "Type in next user to train, or press Recognize"

