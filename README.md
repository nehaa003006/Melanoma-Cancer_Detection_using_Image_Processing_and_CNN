A MATLAB-based system for detecting melanoma and other skin cancers from dermoscopic images using advanced image processing techniques and Convolutional Neural Networks.

🔬 Overview
Skin cancer, particularly melanoma, is a significant public health concern globally. Early detection is crucial for effective treatment and improved survival rates. This project implements an automated melanoma detection system using:

Image Processing techniques for preprocessing and segmentation
Convolutional Neural Networks (CNN) for classification
GLCM-based feature extraction for texture analysis
ABCD rule implementation for melanoma characteristics

The system achieves approximately 95% accuracy on training data and 85-95% accuracy on test data.

Features
Image Processing

--Gaussian filtering for noise reduction
--Grayscale conversion
--Hair removal from dermoscopic images
--Adaptive histogram equalization
--Otsu's thresholding for segmentation
--Morphological operations

Feature Extraction

--GLCM (Gray Level Co-occurrence Matrix) texture features

-Contrast
-Correlation
-Energy
-Homogeneity


Color features (RGB statistics)
Shape descriptors (Area, Perimeter, Eccentricity, etc.)
ABCD melanoma features

Asymmetry
Border irregularity
Color variation
Diameter



Classification

--Deep CNN with 4 convolutional blocks
--Batch normalization
--ReLU activation functions
--Dropout regularization (0.3)
--Adam optimizer

Performance Metrics

--Accuracy
--Sensitivity/Recall
--Specificity
--Precision
--F1-Score
--Confusion Matrix
--ROC Curve and AUC


SYSTEM ARCHITECTURE
Input Image
    ↓
Preprocessing (Gaussian Filter + Grayscale)
    ↓
Segmentation (Otsu's Thresholding)
    ↓
Feature Extraction (GLCM + Shape + Color)
    ↓
CNN Classification
    ↓
Result (Melanoma / Benign)

Software Requirements

--MATLAB R2020a or later
--Image Processing Toolbox
--Deep Learning Toolbox
--Computer Vision Toolbox
--Statistics and Machine Learning Toolbox


======================================================================================================
EXPECTED OUTPUT

Extracted Features:
Contrast: 0.1234
Correlation: 0.8765
Energy: 0.3456
Homogeneity: 0.9123

Classification Results:
Predicted Class: melanoma
=======================================================================================================



Key papers and resources:

Esteva et al. - "Dermatologist-level classification of skin cancer with deep neural networks"
Codella et al. - "Skin lesion analysis toward melanoma detection"
HAM10000 Dataset - "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions"
Confidence Score: 94.57%
