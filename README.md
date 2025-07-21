# NSFW-Content-Detection

This project develops a Convolutional Neural Network (CNN) model to automatically detect Not Safe For Work (NSFW) content in images, primarily for use in identifying and flagging inappropriate content on social media streaming platforms. As in today's world it is a major concern especially for the younger audience.

# Project Overview
With the increasing prevalence of live streaming and user-generated content, social media platforms face significant challenges in moderating inappropriate content in real-time. This project addresses this by providing a machine learning solution to automatically identify and flag NSFW content, enabling platforms to take swift action (e.g., takedown or age-restriction). The model is a deep learning-based image classifier built using TensorFlow/Keras.

# Features
Image Classification: Classifies images as either 'SFW' (Safe For Work) or 'NSFW'.

Convolutional Neural Network (CNN): Utilizes a CNN architecture, well-suited for image recognition tasks.

Data Augmentation: Employs ImageDataGenerator for on-the-fly image augmentation to improve model generalization and robustness.

TensorFlow/Keras: Developed using the powerful and flexible TensorFlow Keras API.

Model Saving: Automatically saves the trained model for future inference.

# Setup and Installation
To set up and run this project, follow these steps:

Clone the repository (if applicable, otherwise save the code):

Bash

git clone <repository_url>
cd <repository_name>
Install dependencies:
Ensure you have Python installed (preferably Python 3.8 or newer). Then, install the required libraries:

Bash

pip install tensorflow pillow
(Pillow is a dependency for PIL.ImageFile and ImageDataGenerator might rely on it.)

Usage
Prepare your dataset according to the structure described in the Dataset section and place it at the path specified by base_dir in the script.

Run the training script:

Bash

python nsfw_detector.py
The script will:

Load and preprocess the image data using ImageDataGenerator.

Build and compile the CNN model.

Train the model for 30 epochs, monitoring validation accuracy and loss.

Evaluate the trained model on the test set.

Save the trained model as cnn_binary_classifier.h5 in the project root directory.

# Model Architecture
The CNN model is a sequential model designed for image binary classification:

Input Layer: Expects images of size (150,150,3) (height, width, color channels).

Convolutional Blocks:

Conv2D with 32 filters, (3,3) kernel, relu activation, followed by MaxPooling2D (2,2).

Conv2D with 64 filters, (3,3) kernel, relu activation, followed by MaxPooling2D (2,2).

Conv2D with 128 filters, (3,3) kernel, relu activation, followed by MaxPooling2D (2,2).

Flatten Layer: Converts the 3D feature maps into a 1D vector.

Dense Layers:

A Dense layer with 128 units and relu activation.

A Dropout layer with a rate of 0.5 for regularization, to prevent overfitting.

A final Dense layer with 1 unit and sigmoid activation, which outputs a probability score for binary classification (e.g., probability of being NSFW).

Training Details
Image Dimensions: 150
times150 pixels.

Batch Size: 32.

Optimizer: Adam optimizer with a learning rate of 0.0001.

Loss Function: Binary Crossentropy (binary_crossentropy), suitable for binary classification.

Metrics: Accuracy.

Epochs: 30.

Data Augmentation (Training Set): Includes image rescaling, horizontal flipping, and zooming.

Data Rescaling (Validation & Test Sets): Only includes image rescaling to normalize pixel values.

Truncated Images: ImageFile.LOAD_TRUNCATED_IMAGES = True is set to handle potentially corrupted or incomplete image files in the dataset without raising an error.

# Evaluation
After training, the model's performance is evaluated on the dedicated test1 dataset.
The script will print the final test accuracy.

# Results
Upon successful execution, the script will:

Show information about the number of images found in each data generator (training, validation, test).

Display training progress per epoch (loss, accuracy for both training and validation sets).

Print the final test accuracy.

Save the trained model as cnn_binary_classifier.h5.

# Future Enhancements
Advanced Architectures: Explore transfer learning with pre-trained models (e.g., VGG, ResNet, EfficientNet) for potentially higher accuracy, especially with larger and more diverse datasets.

Hyperparameter Tuning: Systematically optimize parameters like learning rate, batch size, and network architecture through techniques like Grid Search or Random Search.

Robust Evaluation Metrics: Implement more detailed evaluation metrics such as Precision, Recall, F1-score, ROC curves, and Confusion Matrices to gain deeper insights into model performance for each class.

Real-time Inference: Develop an inference script or API for real-time detection on new images or video streams.

Dataset Expansion & Diversity: Train on a larger and more varied dataset to improve generalization across different types of NSFW content.

Grad-CAM/Saliency Maps: Implement techniques to visualize which parts of the image the CNN focuses on for classification, aiding in interpretability.

Deployment Integration: Outline steps for integrating the model into a social media platform's content moderation pipeline.

# Contact
For any questions or inquiries, please contact me at alanvarghese852@gmail.com.
