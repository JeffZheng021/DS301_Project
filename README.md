## Emotion Recognition from Facial Expressions

## Overview

This project implements a real-time emotion detection system that captures a photo from a webcam and recognizes human emotions from facial expressions. The model has been specifically trained to detect four emotions: Angry, Happy, Sad, and Neutral.

The emotion detection model is trained on a dataset containing images of human faces labeled with one of the four target emotions. The training includes data augmentation like random horizontal flips and random rotations. The model architecture used is a convolutional neural network.

## Model Structure

![366163638_620240163426199_4840107137002992002_n](https://github.com/JeffZheng021/DS301_Project/assets/118134070/140634ff-adf1-4cbe-a1ec-a92ab8090d05)


## Dependencies

PyTorch

OpenCV

NumPy

IPython

torchvision

## Helpful Functions

load_data(): Function for loading and pre-processing the dataset, including data augmentation techniques.

preprocess(): Function for resizing and normalizing the face region of interest to the required input size of the emotion detection model.

predict_emotion(): Function for making emotion predictions on the preprocessed face region.

process_frame(): Function for processing a frame captured from the webcam, including face detection and emotion prediction.

take_photo(): Function that captures a photo using a webcam, allowing the user to click the "Capture" button.

## Results

The model effectively recognizes the emotions Angry, Happy, Sad, and Neutral from facial expressions in real-time.

## Detailed Description

We've used the FER2013 dataset available on kaggle. The dataset consists of 7 classes:
Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral
Here is augmented photos:

<img width="570" alt="Screenshot 2023-08-13 at 11 47 33 PM" src="https://github.com/JeffZheng021/DS301_Project/assets/118134070/2cc8f478-f3ea-41de-8d14-8612f540ddf7">


Model Result:
<img width="916" alt="Screenshot 2023-08-13 at 11 49 35 PM" src="https://github.com/JeffZheng021/DS301_Project/assets/118134070/c2fa05bc-d069-4c71-a039-c0ce84e609c9">

However the result is not as we expected, which is low, so there is another model then trained with 4 classes:
Happy, Sad, Angry, Neutral(this is just added for the real-life capture part)
Accuracy of the model on the test images then jumps to  69.83883346124328%

Model Result:
<img width="921" alt="Screenshot 2023-08-13 at 11 50 07 PM" src="https://github.com/JeffZheng021/DS301_Project/assets/118134070/56ddde66-165d-4709-8c1b-ddc26894a808">

Prediction Showcase:

<img width="837" alt="Screenshot 2023-08-13 at 11 52 21 PM" src="https://github.com/JeffZheng021/DS301_Project/assets/118134070/491e155f-90c3-43e2-ac4f-56908164fe64">



## How to Run

Load Pre-trained Models: Make sure that the pre-trained face detection model and emotion detection model are loaded.

Capture Photo: Use the take_photo() function to capture a photo using the webcam.

Detect Emotions: The captured photo will be processed to detect faces, and the emotions on the faces will be predicted.

**NOTE** The detection might sometimes failed to predict, the reason is due to the resizing and such from the webcam capture, not the actual model's deficiency. If it didn't work the first time, try to re-capture and test it again.






