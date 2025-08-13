% Load the data for forecasting
load authorSamples.mat
load forecastingVocab.mat

% Training
txtTrain = tAusten(1:end-1);
XTrain = dummifyText(txtTrain,vocab);

txtOutput = tAusten(2:end);
YTrain = dummifyText(txtOutput,vocab);

% This network takes about 2+ days to train - we don't have that time
load generateAustenNet.mat

% Create some starter text
startTxt = 'Deep Learning';
startVar = dummifyText(startTxt,vocab);

% Forecast + update state
[net,predicted] = predictAndUpdateState(net,startVar);

% Compute the next character
[~,idx] = max(predicted(:,end));
genChar = char(vocab(idx));
genTxt = [startTxt,genChar];

% Continue generation
for n = 1:200
    genVar = dummifyText(genChar,vocab);
    [net,predicted] = predictAndUpdateState(net,genVar);
    [~,idx] = max(predicted(:,end));
    genChar = char(vocab(idx));
    genTxt = [genTxt,genChar];
end