% FEATURE EXTRACTION MODULE
% This module extracts various features from segmented lesions

function features = extract_features(img, segmented_img)
    % EXTRACT_FEATURES Extracts comprehensive features from skin lesion
    %   Input: img - original RGB image
    %          segmented_img - binary segmented lesion
    %   Output: features - structure containing all extracted features
    
    %% GLCM Features (Texture)
    gray_img = rgb2gray(img);
    
    % Apply mask to focus on lesion region
    masked_img = gray_img;
    masked_img(~segmented_img) = 0;
    
    % Create GLCM with multiple offsets for rotation invariance
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(masked_img, 'Offset', offsets, 'Symmetric', true);
    
    % Extract GLCM properties
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    features.GLCM_Contrast = mean(stats.Contrast);
    features.GLCM_Correlation = mean(stats.Correlation);
    features.GLCM_Energy = mean(stats.Energy);
    features.GLCM_Homogeneity = mean(stats.Homogeneity);
    
    %% Color Features
    % Extract RGB channels
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    
    % Mean color values in lesion region
    features.Color_R_Mean = mean(R(segmented_img));
    features.Color_G_Mean = mean(G(segmented_img));
    features.Color_B_Mean = mean(B(segmented_img));
    
    % Standard deviation of colors
    features.Color_R_Std = std(double(R(segmented_img)));
    features.Color_G_Std = std(double(G(segmented_img)));
    features.Color_B_Std = std(double(B(segmented_img)));
    
    % Color variance
    features.Color_Variance = var(double(gray_img(segmented_img)));
    
    %% Shape Features
    % Get region properties
    props = regionprops(segmented_img, 'Area', 'Perimeter', 'Eccentricity', ...
        'MajorAxisLength', 'MinorAxisLength', 'Solidity', 'Extent');
    
    if ~isempty(props)
        features.Shape_Area = props(1).Area;
        features.Shape_Perimeter = props(1).Perimeter;
        features.Shape_Eccentricity = props(1).Eccentricity;
        features.Shape_MajorAxisLength = props(1).MajorAxisLength;
        features.Shape_MinorAxisLength = props(1).MinorAxisLength;
        features.Shape_Solidity = props(1).Solidity;
        features.Shape_Extent = props(1).Extent;
        
        % Compactness
        features.Shape_Compactness = (props(1).Perimeter^2) / (4 * pi * props(1).Area);
        
        % Circularity
        features.Shape_Circularity = (4 * pi * props(1).Area) / (props(1).Perimeter^2);
        
        % Aspect Ratio
        features.Shape_AspectRatio = props(1).MajorAxisLength / props(1).MinorAxisLength;
    end
    
    %% ABCD Features (Asymmetry, Border, Color, Diameter)
    % These are specific to melanoma detection
    
    % Asymmetry
    features.ABCD_Asymmetry = calculate_asymmetry(segmented_img);
    
    % Border Irregularity
    features.ABCD_BorderIrregularity = calculate_border_irregularity(segmented_img);
    
    % Color Variation
    features.ABCD_ColorVariation = calculate_color_variation(img, segmented_img);
    
    % Diameter (in pixels)
    features.ABCD_Diameter = sqrt(4 * features.Shape_Area / pi);
    
    %% Statistical Features
    lesion_pixels = gray_img(segmented_img);
    
    features.Stat_Mean = mean(lesion_pixels);
    features.Stat_Median = median(lesion_pixels);
    features.Stat_StdDev = std(double(lesion_pixels));
    features.Stat_Variance = var(double(lesion_pixels));
    features.Stat_Entropy = entropy(masked_img);
    features.Stat_Skewness = skewness(double(lesion_pixels));
    features.Stat_Kurtosis = kurtosis(double(lesion_pixels));
    
    %% Edge Features
    edges = edge(gray_img, 'Canny');
    edge_density = sum(edges(:) & segmented_img(:)) / sum(segmented_img(:));
    features.Edge_Density = edge_density;
end

%% Helper Function: Calculate Asymmetry
function asymmetry = calculate_asymmetry(binary_img)
    % Find centroid
    props = regionprops(binary_img, 'Centroid');
    centroid = props(1).Centroid;
    
    % Flip image horizontally and vertically
    flipped_h = flip(binary_img, 2);
    flipped_v = flip(binary_img, 1);
    
    % Calculate asymmetry as XOR difference
    diff_h = xor(binary_img, flipped_h);
    diff_v = xor(binary_img, flipped_v);
    
    asymmetry = (sum(diff_h(:)) + sum(diff_v(:))) / (2 * sum(binary_img(:)));
end

%% Helper Function: Calculate Border Irregularity
function irregularity = calculate_border_irregularity(binary_img)
    % Get boundary
    boundary = bwboundaries(binary_img, 'noholes');
    
    if isempty(boundary)
        irregularity = 0;
        return;
    end
    
    boundary = boundary{1};
    
    % Calculate perimeter
    perimeter = size(boundary, 1);
    
    % Calculate area
    area = sum(binary_img(:));
    
    % Irregularity based on compactness
    irregularity = (perimeter^2) / (4 * pi * area);
end

%% Helper Function: Calculate Color Variation
function variation = calculate_color_variation(rgb_img, mask)
    % Extract colors in lesion region
    R = rgb_img(:,:,1);
    G = rgb_img(:,:,2);
    B = rgb_img(:,:,3);
    
    % Get color values
    r_vals = R(mask);
    g_vals = G(mask);
    b_vals = B(mask);
    
    % Calculate variation as sum of standard deviations
    variation = std(double(r_vals)) + std(double(g_vals)) + std(double(b_vals));
end

%% Helper Function: Create Feature Vector
function feature_vector = create_feature_vector(features)
    % Convert structure to array
    field_names = fieldnames(features);
    feature_vector = zeros(length(field_names), 1);
    
    for i = 1:length(field_names)
        feature_vector(i) = features.(field_names{i});
    end
end
