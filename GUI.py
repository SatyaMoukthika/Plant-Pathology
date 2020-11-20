#!/usr/bin/env python
# coding: utf-8

# In[9]:


# GUI for plant disease detection
import tkinter
from PIL import Image, ImageTk 
from tkinter.filedialog import askopenfilename
window=tkinter.Tk() # creating window named window with tkinter
window.title("plant disease detection") # main window title
window.geometry("500x510") # main window geometry
#window.iconbitmap("C:/Users/Kolluri Midhun/OneDrive/Documents/address icon.png")
img=Image.open('C:/Users/Kolluri Midhun/OneDrive/Documents/icon.jpg') # loding image file from directories
img = img.resize((100, 100)) # resizeing the inserted image to 100*100
img=ImageTk.PhotoImage(img) # storing the resized image in img variable
label=tkinter.Label(image=img).pack() # adding image to label to show on window
title_label=tkinter.Label(window,text="Welcome to cureit ",fg="red",bg="yellow",font=("Helvetica",15,"bold"),padx=30,pady=15,bd=5).pack(pady=10)
window.configure(bg="lightgray")

# printing the team members name if want
def team_members():
    memb_1=tkinter.Label(text=" G shashank",font=("Helvetica",10,"bold")).grid(row=3,column=0,pady=5)

    memb_2=tkinter.Label(text=" G harsha",font=("Helvetica",10,"bold")).grid(row=4,column=0)
   
    memb_3=tkinter.Label(text=" G Srikanth",font=("Helvetica",10,"bold")).grid(row=5,column=0,pady=5)
    
    memb_4=tkinter.Label(text=" T Fanindra",font=("Helvetica",10,"bold")).grid(row=6,column=0)
# printing the remedies for Apple Scrab disease on apple leaf
def Apple_Scrab():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR APPLE SCRAB DISEASE ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "Remove all leaves dropped from tree in the fall and compost to prevent any diseases surviving in debris \n Application of zinc and fertilizer grade urea in the Fall may be necessary to speed leaf drop, lime should then be added to fallen leaves \n fungicide application may be necessary in areas where leaves remain wet for periods in excess of 9 hours \n fungicides such as copper soaps and Bordeaux mixture should be applied if there is a chance of wet period as soon as leaf tips emerge"
    
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",10,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)
# printing remedies for Black_rot disease on apple leaf
def Black_rot():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR BLACK ROT ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "Remove dead wood,mummified fruit and cankers from trees to reduce spread of disease \n Burn any prunings that have been made from the tree \n Disease can be controlled by applying fungicides from silver tip to harvest"
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",10,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)
# printing remedies for Cedar_apple_rust disease on apple leaf
def Cedar_apple_rust():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR CEDAR APPLE RUST ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "Plant resistant varieties when possible \n Rake up and dispose of fallen leaves and other debris from under trees.\n Check to see if you have juniper species near your apple trees.if they infected remove it.\n if growing susceptible varieties in proximity to red cedar follow a fungicide program"
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",10,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)
# printing that leaf is a is healthy condition
def Leaf_healthy():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    
    
    rem1 = "The leaf is healthy "
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",10,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)

    
    

    disease_window.mainloop()
    
 # printing the remidies for BrownSport disease on paddy leaf    
def BrownSpot():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR BROWN SPORT DISEASE ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "Ensure plants are provided with correct nutrients and avoid water stress \n Chemical seed treatments are effective at reducing the incidence of the disease"
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",15,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)
# printing remedies for hipsa disease on paddy leaf
def Hipsa():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR HISPA DISEASE ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "Avoid over dosage of fertilizer application in the field \n Avoid Dense planting \n Cut the shoot tips with the eggs \n Burn the infected leaves away from paddy \n Go for crop rotation process to break the disease "
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",15,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)
# printing remedies for Leaf Blast disease on paddy leaf
def Leaf_Blast():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR LEAF BLAST DISEASE ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "if disease is not endemic to the region, blast can be controlled by planting resistant rice varieties \n Avoid over-fertilizing crop with nitrogen as this increases the plant's susceptibility to the disease \n Utilize good water management to ensure plants do not suffer from drought stress \n Disease can be effectively controlled by the application of appropriate systemic fungicides, where available"
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",15,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)
# printing that paddy is in healthy condition
def Paddy_leaf_healthy():
    
    window.destroy()
    disease_window=tkinter.Tk()
    disease_window.title("Plant Disease Detection")
    disease_window.geometry("500x510")
    
    def exit():
        disease_window.destroy()
    
    rem = "THE REMEDIES FOR PADDY DISEASE ARE "
    remedies = tkinter.Label(text=rem,font=("Helvetica",10,"bold")).grid(row=0,column=0,pady=20)
    
    rem1 = "The leaf is healthy"
    remedies1 = tkinter.Label(text=rem1,font=("Helvetica",15,"bold"),fg="red").grid(row=1,column=0,pady=20)
    
    team=tkinter.Button(text=" team members",command=team_members,font=("Helvetica",10,"bold")).grid(row=2,column=0,pady=10)
    button = tkinter.Button(text="Exit", command=exit).grid(row=7,pady=10)

    
    
    
# implementation of alexnet and training the model with dataset
def training_code(add):
   
    
    import cv2
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras.layers.normalization import BatchNormalization
    np.random.seed(1000)
    


    alexnetModel = Sequential()

    # 1st Convolutional Layer
    alexnetModel.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), activation='relu'))
    # Max Pooling
    alexnetModel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 2nd Convolutional Layer
    alexnetModel.add(Conv2D(filters=256, kernel_size=(11,11), activation='relu'))
    # Max Pooling
    alexnetModel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 3rd Convolutional Layer
    alexnetModel.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu'))

    # 4th Convolutional Layer
    alexnetModel.add(Conv2D(filters=384, kernel_size=(3,3),activation='relu'))

    # 5th Convolutional Layer
    alexnetModel.add(Conv2D(filters=256, kernel_size=(3,3),activation='relu'))
    # Max Pooling
    alexnetModel.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Connected layer
    alexnetModel.add(Flatten())
    # 1st Fully Connected Layer
    alexnetModel.add(Dense(4096, input_shape=(224*224*3,),activation='relu'))
    # Add Dropout to prevent overfitting
    alexnetModel.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    alexnetModel.add(Dense(4096,activation='relu'))
    # Add Dropout
    alexnetModel.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    alexnetModel.add(Dense(1000,activation='relu'))
    # Add Dropout
    alexnetModel.add(Dropout(0.4))


    # Output Layer
    alexnetModel.add(Dense(4,activation='softmax'))

    alexnetModel.summary()

    # Compile the model
    alexnetModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    path = "C:/Users/Kolluri Midhun/OneDrive/Documents/mlsmall/data/"
    output = pd.read_csv(path+"Apple.csv")
    output.head()
    X = []

    for image in glob.glob(path+"Apple___Apple_scab/*.JPG"):
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        X.append(image)

    for image in glob.glob(path+"Apple___Black_rot/*.JPG"):
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        X.append(image)

    for image in glob.glob(path+"Apple___Cedar_apple_rust/*.JPG"):
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        X.append(image)

    for image in glob.glob(path+"Apple___healthy/*.JPG"):
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        X.append(image)
    X = np.array(X)
    X = X.astype("float32")/255.0
    singleDisease = alexnetModel
    singleDisease.fit(X,output,epochs=1)
    
   
        
    #add the input image here, resize the image to shape 1,224,224,3
    lis=[]
    #image =Image.open(filename)
    image = cv2.imread(add)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    lis.append(image)
    image=np.array(lis)
    image = image.astype("float32")/255.0
    y_pred = singleDisease.predict(image)
    #output_label=tkinter.Label(text=str(y_pred[0])).pack()
    #window.destroy()
    analysis=tkinter.Tk()
    analysis.title("plant disease predection")
    analysis.geometry("500x510")
    
    label=tkinter.Label(analysis,text="Apple___Apple_scab disease Accuracy is").pack()
    label=tkinter.Label(analysis,text=str(y_pred[0][0])).pack()
    label=tkinter.Label(analysis,text="Apple___Black_rot disease Accuracy is").pack()
    label=tkinter.Label(analysis,text=str(y_pred[0][1])).pack()
    label=tkinter.Label(analysis,text="Apple___Cedar_apple_rust disease Accuracy is").pack()
    label=tkinter.Label(analysis,text=str(y_pred[0][2])).pack()
    label=tkinter.Label(analysis,text="Apple___healthy disease Accuracy is").pack()
    label=tkinter.Label(analysis,text=str(y_pred[0][3])).pack()
    
# predicting the disease for Apple leaf  from pretrained model       
def  Apple_loadmodel():
    from keras.models import load_model
    import cv2
    from keras.activations import sigmoid


    import numpy as np
    

    path="C:/Users/Kolluri Midhun/OneDrive/Documents/Ai and Ml and Dl/Ai and Ml and Dl/data/Apple/"
    model = load_model(path+"apple.h5")
    
   
    lis=[]
    #image =Image.open(filename)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    lis.append(image)
    image=np.array(lis)
    image = image.astype("float32")/255.0
    y_pred = model.predict(image)
    z = np.exp(y_pred)
    s= np.sum(z)
    output=z/s
    output=output[0]
    
    if(output[0]==max(output)):
        label=tkinter.Label(text="The disease is Apple Scrab",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remidies",command=Apple_Scrab,font=("Helvetica",15,"bold")).pack(pady=10)
    elif(output[1]==max(output)):
        label=tkinter.Label(text="The disease is Black rot",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remidies",command=Black_rot,font=("Helvetica",15,"bold")).pack(pady=10)
    elif(output[2]==max(output)):
        label=tkinter.Label(text="The disease is Cedar apple rust",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remidies",command=Cedar_apple_rust,font=("Helvetica",15,"bold")).pack(pady=10)
    elif(output[3]==max(output)):
        label=tkinter.Label(text="The leaf is healthy",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remedies",command=Leaf_healthy,font=("Helvetica",15,"bold")).pack(pady=10)
        
    else:
        label=tkinter.Label(text="Disease not found",font=("Helvetica",15,"bold")).pack(pady=10)
        button=tkinter.Button(text="team members",command=team_members,font=("Helvetica",15,"bold")).pack(pady=10)
        
def  paddy_loadmodel():
    from keras.models import load_model
    import cv2
    from keras.activations import sigmoid


    import numpy as np
    

    path="C:/Users/Kolluri Midhun/OneDrive/Documents/Ai and Ml and Dl/Ai and Ml and Dl/data/paddy/"
    model = load_model(path+"paddy.h5")
    
   
    lis=[]
    #image =Image.open(filename)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    lis.append(image)
    image=np.array(lis)
    image = image.astype("float32")/255.0
    y_pred = model.predict(image)
    z = np.exp(y_pred)
    s= np.sum(z)
    output=z/s
    output=output[0]
    
    if(output[0]==max(output)):
        label=tkinter.Label(text="The disease is BrownSport",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remidies",command=BrownSpot,font=("Helvetica",15,"bold")).pack(pady=10)
    elif(output[1]==max(output)):
        label=tkinter.Label(text="The leaf is healthy ",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remidies",command=Paddy_leaf_healthy,font=("Helvetica",15,"bold")).pack(pady=10)
    elif(output[2]==max(output)):
        label=tkinter.Label(text="The disease is Hispa",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remidies",command=Hipsa,font=("Helvetica",15,"bold")).pack(pady=10)
    elif(output[3]==max(output)):
        label=tkinter.Label(text="The disease is LeafBlast",font=( "Helvetica",15,"bold"),fg="Darkblue").pack(pady=0)
        button=tkinter.Button(text="Click for remedies",command=Leaf_Blast,font=("Helvetica",15,"bold")).pack(pady=10)
        
    else:
        label=tkinter.Label(text="Disease not found",font=("Helvetica",15,"bold")).pack(pady=10)
        button=tkinter.Button(text="team members",command=team_members,font=("Helvetica",15,"bold")).pack(pady=10)
        
# below function is to access apple infected leaves from dataset
def apple():
    global my_image,filename
    filename = askopenfilename(initialdir='C:/Users/Kolluri Midhun/OneDrive/Documents/Ai and Ml and Dl/Ai and Ml and Dl/data', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    
    my_image=ImageTk.PhotoImage(Image.open(filename))
    label1=tkinter.Label(image=my_image).pack(pady=10)
    button=tkinter.Button(text="Analyze the image",command=Apple_loadmodel,font=("Helvetica",10,"bold")).pack(pady=10)
    
# below function to access paddy infected leaves from dataset
def paddy():
    global my_image,filename
    filename = askopenfilename(initialdir='C:/Users/Kolluri Midhun/OneDrive/Documents/Ai and Ml and Dl/Ai and Ml and Dl/data/Paddy', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    image=Image.open(filename)
    image=image.resize((200, 200))
    my_image=ImageTk.PhotoImage(image)
    label1=tkinter.Label(image=my_image).pack(pady=10)
    button=tkinter.Button(text="Analyze the image",command=paddy_loadmodel,font=("Helvetica",10,"bold")).pack(pady=10)
    





# cateigeory of infected leaf apple or paddy
def image():
    Button=tkinter.Button(text=" paddy ",command=paddy,font=("Helvetica",10,"bold")).pack(pady=10)
    button=tkinter.Button(text=" apple ", command=apple,font=("Helvetica",10,"bold")).pack()
    
 


    
    
    
    
    
    


button1=tkinter.Button(text="Select  leaf category",command=image,font=("Helvetica",10,"bold")).pack() 

window.mainloop()


# In[ ]:




