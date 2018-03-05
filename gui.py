from Tkinter import *
import os

root = Tk(className = 'face_recognition_gui')
root.title('Face Recognizer');
svalue = StringVar() # defines the widget state as string

l = Label(root, text="Add new person")
l.config(font=("Courier", 30))
l.pack()

w = Entry(root,textvariable=svalue) # adds a textarea widget
w.pack()

def train_fisher_btn_load():
    name = svalue.get()
    os.system('python train_fisher.py %s'%name)

def train_eigen_btn_load():
    name = svalue.get()
    os.system('python train_eigen.py %s'%name)

def train_lbph_btn_load():
    name = svalue.get()
    os.system('python train_lbph.py %s'%name)

def recog_fisher_btn_load():
    os.system('python recog_fisher.py')

def recog_eigen_btn_load():
    os.system('python recog_eigen.py')

def recog_lbph_btn_load():
    os.system('python recog_lbph.py')

def recog_dlib():
    os.system('python face_recog.py')

def add_person():
    name = svalue.get()
    os.system('python add_person.py %s'%name)

add_btn = Button(root,text="Add", command=add_person)
add_btn.pack()

f=Frame(root,height=1, width=400, bg="black")
f.pack()

l = Label(root, text="Train")
l.config(font=("Courier", 30))
l.pack()

trainF_btn = Button(root,text="Train (FisherFaces)", command=train_fisher_btn_load)
trainF_btn.pack()

trainE_btn = Button(root,text="Train (EigenFaces)", command=train_eigen_btn_load)
trainE_btn.pack()

recogL_btn = Button(root,text="Train (LBPH)", command=train_lbph_btn_load)
recogL_btn.pack()

f=Frame(root,height=1, width=400, bg="black")
f.pack()

l = Label(root, text="Recognize")
l.config(font=("Courier", 30))
l.pack()

recogF_btn = Button(root,text="Recognize (FisherFaces)", command=recog_fisher_btn_load)
recogF_btn.pack()

recogE_btn = Button(root,text="Recognize (EigenFaces)", command=recog_eigen_btn_load)
recogE_btn.pack()

recogL_btn = Button(root,text="Recognize (LBPH)", command=recog_lbph_btn_load)
recogL_btn.pack()

recogDL_btn = Button(root,text="Recognize (dlib - Deep Learning)", command=recog_dlib)
recogDL_btn.pack()

root.mainloop()
