% Import land cover data
load satData.mat

% Extract some sample images
I = XTrain(:,:,:,800);
RGB = I(:,:,1:3);
NIR = I(:,:,4);
% imshow(RGB)

% Setup training options (again, will cover how we get these later)
opts = trainingOptions('sgdm','MaxEpochs',5,'InitialLearnRate',0.0001);

% Train the network from scratch!
% landnet = trainnet(XTrain,YTrain,net_1,'crossentropy',opts);
load landnet.mat

% Evaluate the network
XTest = double(XTest);
predicted = predict(landnet,XTest);
cats = categories(YTest);
predicted = onehotdecode(predicted,cats,2);
actual = YTest;
confusionchart(actual,predicted)
