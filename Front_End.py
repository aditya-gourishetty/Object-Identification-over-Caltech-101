#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 as cv
import numpy as np
import tensorflow
import tflearn
import os
import matplotlib.pyplot as plt


# In[3]:


adr='C:/Users/Aditya/Desktop/ExpH_Pune/Folder/datasets/VEHICLES/'


# ## Airplanes

# In[4]:


airlist=os.listdir(adr+'airplane')


# In[5]:


airlabel=np.zeros((len(airlist),1))
airlabel.shape


# In[6]:


airfeatures=[]
for a in airlist:
    pln=cv.imread(adr+'airplane/'+a)
    pln=cv.resize(pln,(100,100))
    airfeatures.append(pln)


# In[7]:


airfeatures=np.array(airfeatures)
airfeatures.shape


# ## Car

# In[8]:


carlist=os.listdir(adr+'Car')


# In[9]:


carlabel=np.ones((len(carlist),1))
carlabel.shape


# In[10]:


carfeature=[]
for i in carlist:
    cr=cv.imread(adr+'car/'+i)
    cr=cv.resize(cr,(100,100))
    carfeature.append(cr)


# In[11]:


carfeature=np.array(carfeature)
carfeature.shape


# ## Bike

# In[12]:


bklist=os.listdir(adr+'Bikes')


# In[13]:


bklabel=np.ones((len(bklist),1))*2
bklabel.shape


# In[14]:


bkfeatures=[]
for i in bklist:
    bik=cv.imread(adr+'Bikes/'+i)
    bik=cv.resize(bik,(100,100))
    bkfeatures.append(bik)


# In[15]:


bkfeatures=np.array(bkfeatures)
bkfeatures.shape


# ## Data Combination

# In[16]:


Labels=np.concatenate((airlabel,carlabel,bklabel))
Labels.shape


# In[17]:


Output_Y=[]
for i in Labels:
    if int(i)==0:
        Output_Y.append([1,0,0])
    elif int(i)==1:
        Output_Y.append([0,1,0])
    elif int(i)==2:
        Output_Y.append([0,0,1])
    else:
        print(i)


# In[18]:


Output_Y=np.array(Output_Y)
Output_Y=Output_Y.reshape(-1,3)
Output_Y.shape


# In[19]:


Features=np.concatenate((airfeatures,carfeature,bkfeatures))
Features.shape


# In[20]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Features,Output_Y)


# In[21]:


print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
#ytest


# In[22]:


from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression


# In[23]:


#Input Layer
cnet=input_data(shape=[None,100,100,3],name='input')

#1st Conv. Layer
cnet=conv_2d(cnet,93,8,activation='relu')
#Max-Pool
cnet=max_pool_2d(cnet,5)

#2nd Conv. Layer
cnet=conv_2d(cnet,86,8,activation='relu')
#Max-Pool
cnet=max_pool_2d(cnet,8)

#3rd Conv. Layer
cnet=conv_2d(cnet,79,8,activation='relu')
#Max-Pool
cnet=max_pool_2d(cnet,8)

#4th Conv. Layer
cnet=conv_2d(cnet,72,8,activation='relu')
#Max-Pool
cnet=max_pool_2d(cnet,8)

#Fully Connected
cnet=fully_connected(cnet,(65*65),activation='relu')

#dropout
cnet=dropout(cnet,0.6)

#Output
cnet=fully_connected(cnet,3,activation='softmax')


# In[24]:


cnet=regression(cnet,optimizer='adam',loss='categorical_crossentropy',name='output',learning_rate=0.0003)


# In[25]:


cnnmodel=tflearn.DNN(cnet)


# In[26]:


cnnmodel.fit({'input':xtrain},{'output':ytrain},n_epoch=5,validation_set=({'input':xtest},{'output':ytest}),show_metric=True)


# In[45]:

'''
p_addr='C:/Users/Aditya/Desktop/ExpH_Pune/Folder/ImagePred/'
p_list=os.listdir(p_addr)
predimage=cv.imread(p_addr+'bike4.jpg')

plt.imshow(predimage)
plt.show()
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

'''
from tkinter import *
#import tkinter.messagebox
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

'''
btn1.pack(LEFT)
btn2.pack(LEFT)
tn3.pack(LEFT)
'''
m.mainloop()





