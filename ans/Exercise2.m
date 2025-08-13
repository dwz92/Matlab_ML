% 1. View a spectrogram of a flute playing
I = imread('datasets\Spectrograms\Flute\img1.jpg');
% imshow(I) % one of cello, flute, or piano

% 2. Classify this using AlexNet
net = alexnet;
alexLayers = net.Layers;
predicted = classify(net,I);

% 3. Split into 80/20
imgds = imageDatastore('datasets\Spectrograms', ...
                       'IncludeSubfolders', true, ...
                       'LabelSource','FolderNames');
[trainData,testData] = splitEachLabel(imgds,0.8,'randomized');
trainImages = augmentedImageDatastore([227 227],trainData);
testImages = augmentedImageDatastore([227 227],testData);

% 4. Modify the layers
xferLayers = alexLayers;
xferLayers(23) = fullyConnectedLayer(3,'Name','fc8');
xferLayers(end) = classificationLayer('Name','output');

% 5. Create training options and train
opts = trainingOptions('adam','InitialLearnRate',0.0001);
specnet = trainNetwork(trainImages,xferLayers,opts);

% 6. Evaluate
predicted = classify(specnet,testImages);
actual = testData.Labels;
confusionchart(actual,predicted)