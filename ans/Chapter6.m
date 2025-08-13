% Normal image
I = imread('dog.jpg');
% imshow(I)

% Color distorted image
load trainingDataColor.mat

% Set up the file paths to match
trainingData.File = strcat('datasets/ColorCast/',trainingData.File);

% Import and display a sample distorted image
D = imread(trainingData.File{25});
% imshow(D)

% Transfer learning is available - let's try a DAG net!
net = resnet18;
resLayers = layerGraph(net);

% Replace the appropriate layers
fc = fullyConnectedLayer(3,'Name','fc3');
xferLayers = replaceLayer(resLayers,'fc1000',fc);

rg = regressionLayer('Name','reg');
xferLayers = replaceLayer(xferLayers,'ClassificationLayer_predictions',rg);

xferLayers = removeLayers(xferLayers,'prob');
xferLayers = connectLayers(xferLayers,'fc3','reg');

% Training options (imagine we did what we did in the last chapter)
opts = trainingOptions('adam','InitialLearnRate',0.001, ...
                       'MaxEpochs',5,'Plots','Training-Progress');

% Train the network
% ccnet = trainNetwork(trainingData,xferLayers,opts);
load ccnet.mat

% Evaluate the network
load testData.mat
testData.File = strcat('datasets/ColorCast/',testData.File);
D = imread(testData.File{11});
% imshow(D)

% Test the color corrector
predicted = predict(ccnet,testData);
actual = testData.Color;
err = abs(predicted - actual);

% Regression plot (equiv of a confusion chart for classification)
p = predicted(:,3);
a = actual(:,3);
% plot(p,a,'o')
% hold on
% plot([-60,60],[-60,60])
% hold off

% Actually use the corrector
rgb = predict(ccnet,D);
I = correctColor(D,rgb);
imshowpair(D,I,'montage')