% MAIN PROGRAM - CNN Training for Skin Cancer Detection
% This program trains the CNN model for melanoma detection

clc;
clear all;
close all;

%% Load Dataset
fprintf('Loading dataset...\n');

% Define image datastore
imds = imageDatastore('dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display dataset information
labelCount = countEachLabel(imds);
disp(labelCount);

%% Split Dataset into Training and Testing
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Testing images: %d\n', numel(imdsTest.Files));

%% Data Augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3], ...
    'RandXReflection', true, ...
    'RandYReflection', true);

%% Preprocessing
inputSize = [224 224 3];

augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augImdsTest = augmentedImageDatastore(inputSize, imdsTest);

%% Define CNN Architecture
layers = [
    % Input Layer
    imageInputLayer(inputSize, 'Name', 'input')
    
    % First Convolutional Block
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    % Second Convolutional Block
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    % Third Convolutional Block
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    % Fourth Convolutional Block
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')
    
    % Fully Connected Layers
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu5')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'relu6')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    % Output Layer
    fullyConnectedLayer(numel(categories(imdsTrain.Labels)), 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsTest, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% Train the Network
fprintf('Starting training...\n');
net = trainNetwork(augImdsTrain, layers, options);

%% Save the Trained Model
save('trained_cnn_model.mat', 'net');
fprintf('Model saved successfully!\n');

%% Evaluate the Model
fprintf('Evaluating model...\n');

% Predict on test set
YPred = classify(net, augImdsTest);
YTest = imdsTest.Labels;

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

%% Calculate Performance Metrics
% Sensitivity, Specificity, Precision, F1-Score
cm = confusionmat(YTest, YPred);

% For binary classification
if size(cm, 1) == 2
    TP = cm(1,1);
    TN = cm(2,2);
    FP = cm(2,1);
    FN = cm(1,2);
    
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    precision = TP / (TP + FP);
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
    
    fprintf('\nPerformance Metrics:\n');
    fprintf('Sensitivity (Recall): %.4f\n', sensitivity);
    fprintf('Specificity: %.4f\n', specificity);
    fprintf('Precision: %.4f\n', precision);
    fprintf('F1-Score: %.4f\n', f1_score);
end

%% Plot ROC Curve (for binary classification)
if numel(categories(YTest)) == 2
    scores = predict(net, augImdsTest);
    [X, Y, T, AUC] = perfcurve(YTest, scores(:,1), categories(YTest)(1));
    
    figure;
    plot(X, Y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve (AUC = ', num2str(AUC), ')']);
    grid on;
end

fprintf('\nTraining completed successfully!\n');
