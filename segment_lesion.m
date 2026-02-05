% IMAGE SEGMENTATION MODULE
% This module performs segmentation on preprocessed images

function [segmented_img, boundary] = segment_lesion(preprocessed_img)
    % SEGMENT_LESION Segments the skin lesion from background
    %   Input: preprocessed_img - grayscale preprocessed image
    %   Output: segmented_img - binary segmented image
    %           boundary - boundary coordinates of the lesion
    
    %% Otsu's Thresholding
    % Calculate optimal threshold
    level = graythresh(preprocessed_img);
    
    % Apply threshold with adjustment (0.5 factor mentioned in report)
    adjusted_level = level * 0.5;
    bw = imbinarize(preprocessed_img, adjusted_level);
    
    %% Morphological Operations
    % Remove small objects
    bw_clean = bwareaopen(bw, 50);
    
    % Fill holes
    bw_filled = imfill(bw_clean, 'holes');
    
    % Morphological closing to smooth boundaries
    se = strel('disk', 5);
    bw_closed = imclose(bw_filled, se);
    
    % Morphological opening to remove thin protrusions
    se_open = strel('disk', 3);
    segmented_img = imopen(bw_closed, se_open);
    
    %% Extract Largest Connected Component
    % Label connected components
    cc = bwconncomp(segmented_img);
    numPixels = cellfun(@numel, cc.PixelIdxList);
    
    % Keep only the largest component (assumed to be the lesion)
    [~, idx] = max(numPixels);
    segmented_img = false(size(segmented_img));
    segmented_img(cc.PixelIdxList{idx}) = true;
    
    %% Extract Boundary
    boundary = bwboundaries(segmented_img, 'noholes');
    if ~isempty(boundary)
        boundary = boundary{1};
    else
        boundary = [];
    end
end

%% Helper Function: Advanced Segmentation using Active Contours
function segmented_img = segment_active_contour(preprocessed_img)
    % Initialize mask
    mask = false(size(preprocessed_img));
    mask(25:end-25, 25:end-25) = true;
    
    % Apply active contour
    segmented_img = activecontour(preprocessed_img, mask, 300);
end

%% Helper Function: Watershed Segmentation
function segmented_img = segment_watershed(preprocessed_img)
    % Compute gradient magnitude
    hy = fspecial('sobel');
    hx = hy';
    Iy = imfilter(double(preprocessed_img), hy, 'replicate');
    Ix = imfilter(double(preprocessed_img), hx, 'replicate');
    gradmag = sqrt(Ix.^2 + Iy.^2);
    
    % Watershed transform
    L = watershed(gradmag);
    segmented_img = L > 0;
end

%% Helper Function: Calculate Segmentation Metrics
function metrics = calculate_segmentation_metrics(segmented_img, ground_truth)
    % Calculate Dice coefficient
    intersection = sum(segmented_img(:) & ground_truth(:));
    metrics.Dice = 2 * intersection / (sum(segmented_img(:)) + sum(ground_truth(:)));
    
    % Calculate Jaccard index
    union = sum(segmented_img(:) | ground_truth(:));
    metrics.Jaccard = intersection / union;
    
    % Calculate sensitivity and specificity
    TP = sum(segmented_img(:) & ground_truth(:));
    TN = sum(~segmented_img(:) & ~ground_truth(:));
    FP = sum(segmented_img(:) & ~ground_truth(:));
    FN = sum(~segmented_img(:) & ground_truth(:));
    
    metrics.Sensitivity = TP / (TP + FN);
    metrics.Specificity = TN / (TN + FP);
    metrics.Accuracy = (TP + TN) / (TP + TN + FP + FN);
end
