from tkinter import *
from tkinter import ttk
import cv2 as cv
import numpy as np
m = Tk()
m.title("welcome")
m.minsize(width=500, height=300)
#m.geometry("400x400")
Image=np.array([])
def fun1():
    a=10
    global Image
    print(a)
    vid=cv.VideoCapture(0)

    while(vid.isOpened()):
        r,frame=vid.read()
        cv.imshow('Take Picture',frame)
        key=cv.waitKey(1)

        if key==ord('c'):
            Image=np.array(frame)
            print("Picture Taken")
        if key==ord('q'):
            break
    cv.destroyWindow('Take Picture')

def fun2():
        print(Image.shape)
        cv.imshow('Uploaded',Image)



def fun3():
    print('Checking...')
    predimage=Image
    #plt.imshow(predimage)
    #plt.show()
    predimage=cv.resize(predimage,(100,100))
    predimage=predimage.reshape((1,100,100,3))
    predict_int=np.argmax(cnnmodel.predict(predimage))

    if predict_int==0:
        pred='Airplane'
    elif predict_int==1:
        pred='Car'
    elif predict_int==2:
        pred='Bike'
    print(pred)
label1=ttk.Label(m, text='select the model')
label1.grid(column=1, row= 0, padx=10, pady=10)

btn1= Button(m,text ='Take Picture', command = fun1)
btn1.grid(column =2, row=2 ,padx=10, pady=10)
btn1.config(height= 2, width=30)

btn2 =Button(m,text ='Show Picture',command = fun2)
btn2.grid(column =2,row=5,padx=10,pady=10)
btn2.config(height= 2, width=30)

btn3 =Button(m,text ='Click for Vehicles',command = fun3)
btn3.grid(column =2 ,row=7,padx=10,pady=10)
btn3.config(height=2,width=30)


m.mainloop()





