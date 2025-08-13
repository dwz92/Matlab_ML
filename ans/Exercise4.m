% Load in the data
load fashionDataProcessed.mat

% View some sample images
I = xTest(:,:,:,150);
imshow(I)

% Convert output to non-ordinal
cats = categories(yTrain);
actual = categorical(yTest,cats,'Ordinal',false);

% Setup training options
opts = trainingOptions('adam','MaxEpochs',5);
% fashionnet = trainnet(xTrain,yTrain,net_1,'crossentropy',opts);
load fashionnet.mat

% Evaluate the network
xTest = double(xTest);
predicted = predict(fashionnet,xTest);
cats = categories(yTest);
predicted = onehotdecode(predicted,cats,2);
confusionchart(actual,predicted)