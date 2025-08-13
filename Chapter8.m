load authorSamples.mat
load traindata_s2label.mat

input = XTrain(1);
output = YTrain(1);


%train
XTrain = cellfun(@transpose, XTrain,UniformOutput=false);
opts = trainingOptions('adam', ...
    MaxEpochs=3, ...
    Plots='training-progress');

%net = trainnet(XTrain, YTrain, net_1, 'crossentropy', opts);

load authornet.mat

load testdata_s2label.mat

% test
XTest = cellfun(@transpose, XTest,UniformOutput=false);

temp = zeros(2000, 51, 40);

for n = 1:40
    temp(:,:,n) = XTest{n};
end


predicted = predict(net, temp);

cats = categories(YTest);
predicted = onehotdecode(predicted, cats, 2);
actual = YTest;
%actual = testData.Labels;
confusionchart(actual, predicted)
%accuracy = nnz(predicted == actual) / length(actual) * 100