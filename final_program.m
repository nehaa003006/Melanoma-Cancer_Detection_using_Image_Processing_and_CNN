% FINAL PROGRAM - Skin Cancer Detection
% This is the final program for melanoma cancer detection

% Clear workspace and command window
clc;
clear all;
close all;

% Read the input image
[filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'}, 'Select an image');
if isequal(filename, 0)
    disp('User selected Cancel');
    return;
end

img = imread(fullfile(pathname, filename));

% Display original image
figure;
subplot(2,3,1);
imshow(img);
title('Original Image');

% Convert to grayscale
gray_img = rgb2gray(img);
subplot(2,3,2);
imshow(gray_img);
title('Grayscale Image');

% Apply Gaussian filtering for noise removal
h = fspecial('gaussian', [5 5], 2);
filtered_img = imfilter(gray_img, h);
subplot(2,3,3);
imshow(filtered_img);
title('Gaussian Filtered Image');

% Image segmentation using Otsu's thresholding
level = graythresh(filtered_img);
bw = imbinarize(filtered_img, level);
subplot(2,3,4);
imshow(bw);
title('Segmented Image');

% Morphological operations
se = strel('disk', 5);
bw_clean = imclose(bw, se);
bw_clean = imfill(bw_clean, 'holes');
subplot(2,3,5);
imshow(bw_clean);
title('Morphologically Processed Image');

% Feature extraction using GLCM
glcm = graycomatrix(filtered_img);
stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

% Display extracted features
fprintf('Extracted Features:\n');
fprintf('Contrast: %.4f\n', stats.Contrast);
fprintf('Correlation: %.4f\n', stats.Correlation);
fprintf('Energy: %.4f\n', stats.Energy);
fprintf('Homogeneity: %.4f\n', stats.Homogeneity);

% Load pre-trained CNN model
% Note: Replace with your trained model
load('trained_cnn_model.mat', 'net');

% Prepare image for CNN
img_resized = imresize(img, [224 224]);

% Predict using CNN
[label, scores] = classify(net, img_resized);

% Display results
subplot(2,3,6);
imshow(img);
title(['Prediction: ', char(label)]);

% Display classification results
fprintf('\nClassification Results:\n');
fprintf('Predicted Class: %s\n', char(label));
fprintf('Confidence Score: %.2f%%\n', max(scores)*100);

% Determine if cancerous
if strcmpi(char(label), 'melanoma') || strcmpi(char(label), 'cancerous')
    msgbox('WARNING: Melanoma Detected! Please consult a dermatologist immediately.', 'Detection Result', 'warn');
else
    msgbox('No melanoma detected. Regular monitoring recommended.', 'Detection Result', 'help');
end
