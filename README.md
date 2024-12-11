# Digit Recognizer Using Convolutional Neural Network (CNN)
This project implements a Digit Recognizer using a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model was trained on the Kaggle Digit Recognizer dataset and used for predictions. The main goal is to achieve a high accuracy on digit classification through image processing and deep learning techniques.

# Table of Contents
Project Overview
Data Preprocessing
Model Architecture
Model Training
Evaluation
Prediction and Submission
Requirements
How to Run

## Project Overview

The project uses the MNIST dataset provided in the Kaggle Digit Recognizer competition. The dataset consists of 28x28 pixel images of handwritten digits (0-9). The model uses a Convolutional Neural Network (CNN) to classify these images into their corresponding digit labels.

## Key Features:

Data preprocessing including normalization and reshaping for CNN.
One-hot encoding for categorical labels.
Data augmentation for improving model generalization.
CNN model architecture with two convolutional layers and dropout regularization.
Model training with Adam optimizer and categorical cross-entropy loss function.
Visualization of training accuracy and loss over epochs.
Predictions on test data and generation of a submission file.

## Data Preprocessing

The data is preprocessed in the following steps:

Loading Data: The dataset is loaded from CSV files.
Feature and Label Separation: The features (images) and labels (digit values) are separated.
Normalization: Pixel values are normalized by dividing them by 255 to scale them between 0 and 1.
Reshaping: The data is reshaped to 28x28x1 to be compatible with the input requirements of CNN.
One-hot Encoding: Labels are converted into one-hot encoded format using Keras to_categorical function.

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of the following layers:

Conv2D (First Layer): Convolutional layer with 8 filters, kernel size of 5x5, and ReLU activation.
MaxPooling2D (First Pooling Layer): Reduces the spatial dimensions from (28x28) to (14x14).
Dropout (First Dropout Layer): A dropout layer with 25% probability to avoid overfitting.
Conv2D (Second Layer): Convolutional layer with 16 filters, kernel size of 3x3, and ReLU activation.
MaxPooling2D (Second Pooling Layer): Reduces the dimensions from (14x14) to (7x7).
Dropout (Second Dropout Layer): A dropout layer with 25% probability.
Flatten: Flattens the output to a 1D vector for input to fully connected layers.
Dense (Fully Connected Layer): Fully connected layer with 128 units and ReLU activation.
Dropout (Third Dropout Layer): A dropout layer with 50% probability.
Output Layer: A dense layer with 10 units (one for each digit) and softmax activation for multi-class classification.

## Model Training

The model is compiled with the Adam optimizer and categorical cross-entropy loss. The model is trained using the ImageDataGenerator for data augmentation, which includes rotation, width and height shifts, and zooming. Training runs for 20 epochs.

Batch Size: 64
Epochs: 20
The training history is visualized by plotting the training and validation accuracy and loss over epochs.

## Evaluation

After training, the model is evaluated on the validation set. The accuracy is computed, and the loss and accuracy are plotted for both the training and validation datasets to assess the model's performance.

## Prediction and Submission

The trained model is used to predict the labels for the test data. The predictions are then written to a CSV file in the format required by Kaggle for submission.

The submission CSV file is saved as Submission.csv, containing the ImageId and Label columns.

Requirements
To run this project, you need the following Python libraries:

numpy
pandas
matplotlib
seaborn
keras
tensorflow
scikit-learn














