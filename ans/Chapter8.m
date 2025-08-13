% Import some example data
load authorSamples.mat
load traindata_s2label.mat
input = XTrain{1};
output = YTrain(1);

% This LSTM expects data to be MxN where M = time and N = features
XTrain = cellfun(@transpose,XTrain,'UniformOutput',false);

% Train the network
opts = trainingOptions('adam','MaxEpochs',3,'Plots','Training-Progress');
% net = trainnet(XTrain,YTrain,net_1,'crossentropy',opts);
load authornet.mat

% Evaluate the network
load testdata_s2label.mat
XTest = cellfun(@transpose,XTest,'UniformOutput',false);

% Shape the data into a numerical array
temp = zeros(2000,51,40);
for n = 1:40
    temp(:,:,n) = XTest{n};
end

% Confusion
predicted = predict(net,temp);
cats = categories(YTest);
predicted = onehotdecode(predicted,cats,2);
actual = YTest;
confusionchart(actual,predicted)