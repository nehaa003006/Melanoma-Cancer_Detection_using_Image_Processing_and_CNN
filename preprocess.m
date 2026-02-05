clc;
clear all;
close all;

% Change directory to where the images are stored
cd Database;

% Loop through images in the directory
for i = 1:14
    % Construct the image file name
    st = int2str(i);
    str = strcat(st, '.jpg');

    % Read the image
    I = imread(str);
    Ia = I;

    % Resize and convert the image to double precision
    I = double(imresize(I, [256 256]));

    % Preprocessing
    % Apply Gaussian filter
    g = fspecial('gaussian');
    pre(:,:,1) = imfilter(double(I(:,:,1)), g);
    pre(:,:,2) = imfilter(double(I(:,:,2)), g);
    pre(:,:,3) = imfilter(double(I(:,:,3)), g);

    % Convert to grayscale
    I_gray = rgb2gray(uint8(pre));

    % Create a figure to display the stages
    figure;
    subplot(2,2,1);
    imshow(uint8(Ia));
    title('Original Image');

    subplot(2,2,2);
    imshow(uint8(pre));
    title('Gaussian Filtered');

    subplot(2,2,3);
    imshow(I_gray);
    title('Grayscale Image');

    % More preprocessing steps can be added here as needed

    % Save the figure or display it
    % saveas(gcf, strcat('Preprocessing_Stages_', st, '.png'));
end

% Change back to the original directory
cd ..
