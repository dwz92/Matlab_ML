I = imread("dog.jpg");

load trainingDataColor.mat

trainingData.File = strcat('datasets/ColorCast/', trainingData.File);

D = imread(trainingData.File{1});
%imshow(D)

net = resnet18;
resLayers = layerGraph(net);

fc = fullyConnectedLayer(3, "Name", 'fc3');
xferLayers = replaceLayer(resLayers, 'fc1000', fc);

rg = regressionLayer("Name", 'reg');
xferLayers = replaceLayer(xferLayers, 'ClassificationLayer_predictions', rg);

xferLayers = removeLayers(xferLayers, 'prob');
xferLayers = connectLayers(xferLayers, 'fc3', 'reg');


opts = trainingOptions("adam", "Plots", "training-progress", ...
    "InitialLearnRate", 0.0001, ...
    "MaxEpochs", 5, ...
    "Plots", "training-progress");


% ccnet = trainNetwork(trainingData, xferLayers, opts);

load ccnet.mat

load testData.mat


testData.File = strcat('datasets/ColorCast/' , testData.File);

D = imread(testData.File{11});

predicted = predict(ccnet, testData)
actual = testData.Color
err = abs(predicted - actual)

%red
redPredicted = predicted(:, 1)
redActual = actual(:, 1)

%plot(redActual, redPredicted, 'o')

%hold on 

%plot([-60 60], [-60 60])

%hold off

% green
grePredicted = predicted(:, 2)
greActual = actual(:, 2)

%plot(redActual, redPredicted, 'o')

%hold on 

%plot([-60 60], [-60 60])

%hold off


%blue
bluPredicted = predicted(:, 2)
bluActual = actual(:, 2)


rgb = predict(ccnet, D)
I = correctColor(D, rgb)
imshowpair(D, I, 'montage')
