from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import easygui as eg
#import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from os import path
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import cv2
import os
from os import listdir
from os.path import isfile, join
import tensorflow.keras.backend as K
#Import the VGGNet architecture
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

choices = ["Add a new image", "Train VGGNet", "Recognize using VGGNet", "Exit"]
choice = 0
base_folder = 'database/'
dsize = (128, 128)
rows = dsize[0]
cols = dsize[1]
folder_names = []
for entry_name in os.listdir(base_folder):
    entry_path = os.path.join(base_folder, entry_name)
    if os.path.isdir(entry_path):
        folder_names.append(entry_name)

model_name = "multiple_trained_cnn_vgg162.model"
channels = 3

while(choice != choices[3]) :
    text = "Select an option"
    title = text
    
    choice = eg.choicebox(text, title, choices)
    
    if(choice == choices[0]) :
        file = eg.fileopenbox();
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", type=str,
                        default=file,
            help="path to input image")
        ap.add_argument("-f", "--face", type=str,
            default="face_detector",
            help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
            default="mask_detector.model",
            help="path to trained face detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())
        
        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        # load the face mask detector model from disk
        print("[INFO] loading face detector model...")
        model = load_model(args["model"])
        
        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        image = cv2.imread(args["image"])
        orig = image.copy()
        (h, w) = image.shape[:2]
        
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the face detections
        print("[INFO] computing face detections...")
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
        
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
        
                # pass the face through the model to determine if the face
                # has a mask or not
                (mask, withoutMask) = model.predict(face)[0]
        
                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "No Face" if mask > withoutMask else "Face"
                color = (0, 255, 0) if label == "No Face" else (0, 0, 255)
                
                if(withoutMask > mask) :
                    name = eg.enterbox("Enter person name", "Person name", "p1")
                    if not os.path.exists(base_folder + name):
                        os.mkdir(base_folder + name)
                    output_img = cv2.resize(image, dsize)
                    DIR = base_folder + name
                    counter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
                    cv2.imwrite(base_folder + name + "/" + str(counter) + '.png',output_img)
                    print('Total images for ' + name + ' are ' + str(counter+1))
                    
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    elif(choice == choices[1]) :
        
        train_dir = 'database/'
        test_dir = 'database/'
        validation_dir = 'database/'
        
        train_datagen = ImageDataGenerator(rescale=1.0/255)
        train_batch_size = 32;
        
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        validation_datagen = ImageDataGenerator(rescale=1.0/255)
        
        #Used to get the same results everytime
        np.random.seed(42)
        #tf.random.set_seed(42)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(rows,cols),
            batch_size=train_batch_size,
            class_mode='categorical')
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(rows,cols),
            batch_size=20,
            class_mode='categorical')
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(rows,cols),
            batch_size=20,
            class_mode='categorical')
        
        ######################################################################
        #initialize the NN
        
        #Load the VGG16 model, use the ILSVRC competition's weights
        #include_top = False, means only include the Convolution Base (do not import the top layers or NN Layers)
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(rows,cols,3))
        conv_base.trainable = False;
        model = models.Sequential()
        
        #Add the VGGNet model
        model.add(conv_base)
        
        #NN Layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256,activation='relu'))
        model.add(keras.layers.Dense(len(os.listdir(train_dir)),activation='softmax'))
        
        print(model.summary())
        ######################################################################
        
        #Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        
        #Steps per epoch = Number of images in the training directory / batch_size (of the generator)
        #validation_steps = Number of images in the validation directory / batch_size (of the generator)
        checkpoint_callback = keras.callbacks.ModelCheckpoint("%s" % (model_name),
                                                                  save_best_only=True)
        model_history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs = 10,
            validation_data=validation_generator,
            validation_steps=50,
            callbacks = [checkpoint_callback])
        
        #Plot the model
        pd.DataFrame(model_history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()
        
        model.save(model_name)
        
        #Save the history in CSV file
        hist_csv_file = 'vggnet_history.csv'
        hist_df = pd.DataFrame(model_history.history)
        with open(hist_csv_file,mode='w') as f:
            hist_df.to_csv(f)
    elif(choice == choices[2]) :
        file = eg.fileopenbox();
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", type=str,
                        default=file,
            help="path to input image")
        ap.add_argument("-f", "--face", type=str,
            default="face_detector",
            help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
            default="mask_detector.model",
            help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())
        
        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        model = load_model(args["model"])
        
        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        image = cv2.imread(args["image"])
        orig = image.copy()
        (h, w) = image.shape[:2]
        
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the face detections
        print("[INFO] computing face detections...")
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
        
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
        
                # pass the face through the model to determine if the face
                # has a mask or not
                (mask, withoutMask) = model.predict(face)[0]
        
                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "No Face" if mask > withoutMask else "Face"
                color = (0, 255, 0) if label == "No Face" else (0, 0, 255)
                
                if(withoutMask > mask) :
                    model = keras.models.load_model(model_name)
                    frame = cv2.imread(file,cv2.IMREAD_UNCHANGED)
                    frame = cv2.resize(frame, (rows,cols), interpolation = cv2.INTER_AREA)
                    frame_bkp = frame
                    channels = 3
                    
                    frame = np.asarray(frame).reshape((1,rows,cols,channels))
                    #Convert frame into tensor
                    #frame = K.constant(frame)
                    y_pred = model.predict_classes(frame)
                    y_pred = y_pred[0]
                    label = folder_names[y_pred]
                    
                    output = label
                    
                    print("Classified as %s\n" % (label))
                    cv2.rectangle(frame_bkp, (0, 0), (300, 40), (0, 0, 0), -1)
                    cv2.putText(frame_bkp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
                    cv2.imshow("Face Recognition", frame_bkp)
                    cv2.waitKey(1000)
                    
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        
cv2.destroyAllWindows()