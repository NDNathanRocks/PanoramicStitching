image = rgb2gray(imresize(imread('images/guitar-im1.png'), [300 300]));
image2 = rgb2gray(imresize(imread('images/guitar-im2.png'), [300 300]));
% image = rgb2gray(imresize(imread('images/backyard-im1.png'), [300 300]));
% image2 = rgb2gray(imresize(imread('images/backyard-im2.png'), [300 300]));
% image = rgb2gray(imresize(imread('images/forestwalk-im1.png'), [300 300]));
% image2 = rgb2gray(imresize(imread('images/forestwalk-im2.png'), [300 300]));

fastThres = 0.01;
harrisThres = 0.000005;

image = im2double(image);
image2 = im2double(image2);

FAST_M = my_fast_detector(image, fastThres);
FAST_M2 = my_fast_detector(image2, fastThres);

[matchedPoints1, matchedPoints2] = generateMatch(image, FAST_M, image2, FAST_M2);

[tform, inlierIdx] = estimateGeometricTransform2D(matchedPoints2, matchedPoints1,"similarity");

sizeOfImage = size(image);
[xlim, ylim] = outputLimits(tform, [1 300], [1 300]);   

xMin = min([1 xlim]);
xMax = max([300 xlim]);

yMin = min([1 ylim]);
yMax = max([300 ylim]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port'); 

panorama = zeros([height width], 'like', image);

xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width]);
                 
mask = true( size(image, 1), size(image, 2));

panorama = step(blender, panorama, image, mask);

warpedImage = imwarp(image2, tform, 'OutputView', panoramaView);
              
mask = imwarp(true( size(image2, 1), size(image2, 2) ), tform, 'OutputView', panoramaView);

panorama = step(blender, panorama, warpedImage, mask);

figure
imshow(panorama)

function [matchedPoints1,matchedPoints2] = generateMatch(image, matrix, image2, matrix2)

    points = detectORBFeatures(matrix,'ScaleFactor',1.01,'NumLevels',3);
    points2 = detectORBFeatures(matrix2,'ScaleFactor',1.01,'NumLevels',3);
    
    [ORBfeatures, valid_points1] = extractFeatures(image, points);
    [ORBfeatures2, valid_points2] = extractFeatures(image2, points2);
      
    indexPairs = matchFeatures(ORBfeatures, ORBfeatures2);

    matchedPoints1 = valid_points1(indexPairs(:,1),:);
    matchedPoints2 = valid_points2(indexPairs(:,2),:);
    
    figure;
    showMatchedFeatures(image,image2,matchedPoints1,matchedPoints2,"montag",Parent=axes);

end

function returnvalue = my_fast_detector(image, thres)
    
    M1 = circshift(image, [3 0]);
    M2 = circshift(image, [3 -1]);
    M3 = circshift(image, [2 -2]);
    M4 = circshift(image, [1 -3]);
    M5 = circshift(image, [0 -3]);
    M6 = circshift(image, [-1 -3]);
    M7 = circshift(image, [-2 -2]);
    M8 = circshift(image, [-3 -1]);
    M9 = circshift(image, [-3 0]);
    M10 = circshift(image, [-3 1]);
    M11 = circshift(image, [-2 2]);
    M12 = circshift(image, [-1 3]);
    M13 = circshift(image, [0 3]);
    M14 = circshift(image, [1 3]);
    M15 = circshift(image, [2 2]);
    M16 = circshift(image, [3 1]);
    
    %HIGH SPEED TEST%
    M1_logic = (M1 - thres > image) | (M1 + thres < image);
    M9_logic = (M9 - thres > image) | (M9 + thres < image);

    M1M9_logic = M1_logic & M9_logic;

    M5_logic = (M5 - thres > image) | (M5 + thres < image);
    M13_logic = (M13 - thres > image) | (M13 + thres < image);

    M5M13_logic = M5_logic | M13_logic;

    speed_test_logic = M1M9_logic & M5M13_logic;

    M1_logic = double((M1-thres > image) | (M1 + thres < image));
    M2_logic = double((M2-thres > image) | (M2 + thres < image));
    M3_logic = double((M3-thres > image) | (M3 + thres < image));
    M4_logic = double((M4-thres > image) | (M4 + thres < image));
    M5_logic = double((M5-thres > image) | (M5 + thres < image));
    M6_logic = double((M6-thres > image) | (M6 + thres < image));
    M7_logic = double((M7-thres > image) | (M7 + thres < image));
    M8_logic = double((M8-thres > image) | (M8 + thres < image));
    M9_logic = double((M9-thres > image) | (M9 + thres < image));
    M10_logic = double((M10-thres > image) | (M10 + thres < image));
    M11_logic = double((M11-thres > image) | (M11 + thres < image));
    M12_logic = double((M12-thres > image) | (M12 + thres < image));
    M13_logic = double((M13-thres > image) | (M13 + thres < image));
    M14_logic = double((M14-thres > image) | (M14 + thres < image));
    M15_logic = double((M15-thres > image) | (M15 + thres < image));
    M16_logic = double((M16-thres > image) | (M16 + thres < image));
    
    M_logic = M1_logic + M2_logic + M3_logic + M4_logic + M5_logic + M6_logic + M7_logic + M8_logic + M9_logic + M10_logic + M11_logic + M12_logic + M13_logic + M14_logic + M15_logic + M16_logic;
    M_logic = M_logic >= 8;

    final_logic = speed_test_logic & M_logic;

    %NON MAXIMAL SUPPRESSION%
    v = abs(image-M1) + abs(image-M2) + abs(image-M3) + abs(image-M4) + abs(image-M5) + abs(image-M6) + abs(image-M7) + abs(image-M8) + abs(image-M9) + abs(image-M10) + abs(image-M11) + abs(image-M12) + abs(image-M13) + abs(image-M14) + abs(image-M15) + abs(image-M16);

    temp = double(v) .* double(final_logic);

    downshifted = circshift(final_logic, [1 0]);
    rightshifted = circshift(final_logic, [0 1]);

    res = (temp > downshifted) > rightshifted;

    returnvalue = res;

end