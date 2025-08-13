I = imread("datasets\Spectrograms\Flute\img1.jpg");

%imshow(I);

net = alexnet;
alexlayers = net.Layers;

predicted = classify(net, I);

%Q3

imds = imageDatastore("datasets\Spectrograms\", ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

[traindata, testdata] = splitEachLabel(imds, 0.8, "randomized");

xferlayers = alexlayers;
xferlayers(23) = fullyConnectedLayer(3, 'Name', 'fc8');
xferlayers(end) = classificationLayer("Name",'output');

opts = trainingOptions("adam", "InitialLearnRate", 0.0001);

%insnet = trainNetwork(traindata,xferlayers, opts);

load insnet.mat

% test
predicted = classify(insnet, testdata);
actual = testdata.Labels;
%confusionchart(actual, predicted)
accuracy = nnz(predicted == actual) / length(actual) * 100