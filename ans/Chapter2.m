% Import a pretrained network
net = alexnet;
alexLayers = net.Layers;

% Graph the network structure
% analyzeNetwork(net)

% Extract the input/output layers
in = alexLayers(1);
out = alexLayers(end);
classes = out.Classes;

% Try classifying one of our images
I = imread('Harper.jpg');
% imshow(I)

% Resize to match network input size
I = imresize(I,[227 227]);
% imshow(I)

% Classify this image using AlexNet
[predicted,scores] = classify(net,I);
% bar(scores)

% Classify a batch of images
imgds = imageDatastore('datasets\petImages\Harper');
% montage(imgds)

% Resize all images in this batch
augds = augmentedImageDatastore([227 227],imgds);

% Classify these resized images
predicted = classify(net,augds);

% Transfer learning - modifying a pretrained network
% to match your application needs

% Step 1: Replace specific layers within the network
xferLayers = alexLayers;
xferLayers(23) = fullyConnectedLayer(14,'Name','fc8');
xferLayers(end) = classificationLayer('Name','output');

% Step 2: Prepare data for retraining the network
imgds = imageDatastore('datasets\petImages', ...
                       'IncludeSubfolders', true, ...
                       'LabelSource','FolderNames');

% Split this data into training and testing
[trainData,testData] = splitEachLabel(imgds,0.7,'randomized');

% Make sure these images match the size of the input
trainImages = augmentedImageDatastore([227 227],trainData);
testImages = augmentedImageDatastore([227 227],testData);

% Step 3: Check GPU and prepare training options
opts = trainingOptions('adam','InitialLearnRate',0.0001);

% Step 4: Use 1-3 to retrain your network!
% petnet = trainNetwork(trainImages,xferLayers,opts);
load petnet.mat % holds the retrained network so we don't do this again

% Evaluate the network by classifying the test data
predicted = classify(petnet,testImages);
actual = testData.Labels;
% confusionchart(actual,predicted)
accuracy = nnz(predicted == actual) / length(actual) * 100;