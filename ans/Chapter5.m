% Import the data + network from yesterday
load satData.mat
load satLayers.mat
layers(end) = [];

% To start, try the standard training options
% 1. If your loss increases over time or becomes NaN
% this might be because your learning rate is too high.
% 2. If your loss then converges early, consider tuning
% the number of epochs to be less.
% 3. Alternatively, you can use validation data as a stop criteria!
% 4. After your loss converges, try playing around with the learning
% rate to see if there's any room for your algorithm to learn.
valData = {XVal, YVal};
opts = trainingOptions('rmsprop','Plots','Training-Progress', ...
                       'InitialLearnRate',0.0001, ...
                       'ValidationData',valData, ...
                       'ValidationFrequency',5, ...
                       'ValidationPatience',10, ...
                       'LearnRateSchedule','piecewise', ...
                       'LearnRateDropPeriod',1, ...
                       'LearnRateDropFactor',0.7);
landnet = trainnet(XTrain,YTrain,layers,'crossentropy',opts);

% Evaluate the network
XTest = double(XTest);
predicted = predict(landnet,XTest);
cats = categories(YTest);
predicted = onehotdecode(predicted,cats,2);
actual = YTest;
confusionchart(actual,predicted)
