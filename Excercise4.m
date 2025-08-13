load fashionDataProcessed.mat

I = xTest(:, :, :, 150);
%imshow(I)

opts = trainingOptions("adam");

%fasnet = trainnet(xTrain, yTrain, net_1, 'crossentropy', opts);

load fasnet.mat

xTest = double(xTest);
predicted = predict(fasnet, xTest);

cats = categories(yTest);
predicted = onehotdecode(predicted, cats, 2);
actual = categorical(yTest, cats, 'Ordinal',false);
confusionchart(actual, predicted)