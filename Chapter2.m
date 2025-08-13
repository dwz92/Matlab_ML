%Import Alex Net

net = alexnet;
alexLayers = net.Layers;

%Graph
%analyzeNetwork(net)

in = alexLayers(1);
out = alexLayers(end);
classes = out.Classes;

%Eg
I = imread("harper.jpg");

%Resize
I = imresize(I, [227 227]);
%imshow(I)

%Claasify
[predicted, scores] = classify(net, I);

%bar(scores)

%Classify a batch
imgds = imageDatastore('datasets/petImages/Harper');
%montage(imgds)

augds = augmentedImageDatastore([227 227], imgds)


%Classify again
predicted = classify(net, augds)

% transfer learning, mod a pretrained nn for personal use

%replace some layers
xferlayers = alexLayers;
xferlayers(23) = fullyConnectedLayer(14, 'Name','fc8');
xferlayers(end) = classificationLayer("Name",'output');

%Train nn w data
imgds = imageDatastore("datasets\petImages\", ...
    "IncludeSubfolders",true, ...
    "LabelSource","Foldernames");

% Split ds
[trainData, testData] = splitEachLabel(imgds, 0.7, "randomized");


%match size
trainImages = augmentedImageDatastore([227 227], trainData);
testImages = augmentedImageDatastore([227 227], testData);


%check gpu
%gpuDevice

%train alg
opts = trainingOptions("adam", "InitialLearnRate", 0.0001);

%petnet = trainNetwork(trainImages,xferlayers, opts);


%save it lol
load petnet.mat

% test
predicted = classify(petnet, testImages);
actual = testData.Labels;
%confusionchart(actual, predicted)
accuracy = nnz(predicted == actual) / length(actual) * 100