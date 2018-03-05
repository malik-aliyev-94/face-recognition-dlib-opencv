import numpy as np
import cv2
import sys
import os

class TrainFisherFaces:
    def __init__(self):
        cascPath = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        if not os.path.isdir('trained_data'):
            os.mkdir('trained_data')
        self.model = cv2.face.FisherFaceRecognizer_create()

    def are_enough_faces(self):
        existingFaces = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                existingFaces += 1

        if existingFaces > 1:
            return True
        else:
            return False

    def fisher_train_data(self):
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
        self.model.save('trained_data/fisher_trained_data.xml')
        print "Training completed successfully"
        return


if __name__ == '__main__':
    trainer = TrainFisherFaces()

    if trainer.are_enough_faces():
        trainer.fisher_train_data()
        print "1.Type in next user to train, or press Recognize"
    else:
        print "2.Type in next user to train, or press Recognize"

