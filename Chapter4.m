load satData.mat

I = XTrain(:, :, :, 800);

RGB = I(:, :, 1:3);
NIR = I(:, :, 4);
%imshow(RGB)


%train
opts = trainingOptions("sgdm", 'MaxEpochs',5, 'InitialLearnRate',0.0001);

%landnet = trainnet(XTrain, YTrain, net_1, 'crossentropy', opts);

load landnet.mat

% test
XTest = double(XTest);
predicted = predict(landnet, XTest);

cats = categories(YTest);
predicted = onehotdecode(predicted, cats, 2);
actual = YTest
%actual = testData.Labels;
confusionchart(actual, predicted)
%accuracy = nnz(predicted == actual) / length(actual) * 100