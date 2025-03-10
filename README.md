# Face Recognition Using PCA, LDA, and MLP

## Project Overview
This project implements a face recognition system using Principal Component Analysis (PCA) for feature extraction, Linear Discriminant Analysis (LDA) for dimensionality reduction, and a Multi-Layer Perceptron (MLP) for classification. The dataset consists of face images stored in a directory structure where each subdirectory corresponds to an individual.

## Features
- **Face Dataset Processing**: Reads images from a dataset directory and preprocesses them.
- **PCA (Eigenfaces)**: Extracts principal components from face images to reduce dimensionality.
- **LDA (Fisherfaces)**: Further reduces dimensions while preserving class separability.
- **MLP Classifier**: Uses a neural network for face classification.
- **Performance Evaluation**: Computes the accuracy of the model on the test dataset.

## Dependencies
Ensure the following Python libraries are installed:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

## Dataset Structure
The dataset should be organized as follows:
```
/dataset/faces/
    /person_1/
        image1.jpg
        image2.jpg
    /person_2/
        image1.jpg
        image2.jpg
```

## Steps to Run the Project
1. **Load Dataset**: Images are read from directories and converted to grayscale.
2. **Preprocessing**:
   - Resizing images to a fixed size (300x300 pixels).
   - Flattening images into 1D feature vectors.
3. **Split Data**: The dataset is split into 70% training and 30% testing sets.
4. **Feature Extraction**:
   - Apply PCA to extract eigenfaces.
   - Apply LDA for dimensionality reduction.
5. **Train MLP Classifier**: Train a neural network with two hidden layers.
6. **Make Predictions**: Evaluate the trained model on the test set.
7. **Calculate Accuracy**: Compute accuracy based on true positives.
8. **Visualize Results**: Display sample test images with predicted labels.

## Model Performance
The final accuracy of the model is printed after classification. The model's performance can be improved by tuning hyperparameters and increasing dataset size.

## Visualization
- The eigenfaces are displayed after PCA transformation.
- Test images along with their predicted and true labels are plotted.
