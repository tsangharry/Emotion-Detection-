# Emotion Detection Classifier

This project is an end-to-end machine learning project whereby object recognition is used to classify people's emotion. \
Deployment: Streamlit \
Object Detection: Tensorflow, cv2

<div style="text-align:center"><img src="assets/gif_emotions.gif" | width=1100 /></div>
<br>
<br>

# About the Emotion Detection Classifer

## Background:
The CNN model has been trained using the Tensorflow framework to recognise 5 emotion classes - Happiness, Neutral, Sadness, Angry and Surprised. OpenCV was used to work with video streams and video files. Streamlit was used to build the front end web application.

## Model Architecture:
The CNN model consists of the following layers:
- Convolution (64 Filters)
    - Conv2D
    - Batch Norm 
    - Conv2D
    - Batch Norm
    - Max Pool
    - Dropout
- Convolution (128 Filters)
- Convolution (256 Filters)
- FC Batch Norm Dropout
- Softmax Output 

## Preprocessing and Training
1. Input Images
    - The input images were of size 48 (height) x 48 (width).
    - Colour and Black and White Images
    - Only supports images of format .png, .jpeg and .jpg are currently supported.

2. Data Augmentation
    - Rotation Range: 15 degrees
    - Width Shift: 0.15
    - Height Shift: 0.15
    - Shear range: 0.15
    - Zoom range: 0.15
    - Horizontal Flip: True
    

3. Model Training Parameters
    - Loss function: Categorical Cross Entropy
    - Optimizer: Adam
    - Learning Rate Decay: 0.0001
    - Batch Size: 512
    - Train-Test Split:
        - Training Images: 92,968
        - Validation Images:


4. Performance
    - Test Accuracy: 66%
    
## Folder Structure
```
├── face_detector
    ├──  deploy.protxt
    ├──  res10_300x300_ssd_iter_14000.caffemodel
├── model
    ├── emo.h5
├── notebooks
    ├── eda.ipynb
├── scripts
    ├──  start_app.sh
├── src
    ├──  datapipeline
        ├──  loader.py
    ├──  experiment
        ├──  main.py
    ├──  livestream
        ├──  detect_emotion_video.py
    ├──  modelling
        ├──  model.py
├──  app.py
└── tests
    ├── __init__.py
    └── test_inference.py
├── Dockerfile
├── LICENCE
├── README.md
├── conda.yml
├── init.sh
├── skaffold.yaml
```

# Getting Started
The set of instructions below will get you a copy of the project up and running locally.

## Deployment on local machine
### Step 1:
Clone the repository 

### Step 2:
Create a new environment and install the necessary packages and dependencies from the conda.yml file

```
conda env create -f conda.yml
```

### Step 3: 
Run the detection_emotion_video python file
```
python streamlit run app.py
```

# Authors
- Harry Tsang 
- Joy Sng 
- Alvin Lee 
- Santosh Yadaw 

# Acknowledgements
- Dataset: https://www.kaggle.com/mahmoudima/mma-facial-expression


