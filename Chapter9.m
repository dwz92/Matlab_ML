load authorSamples.mat
load forecastingVocab.mat


txtTran = tAusten(1:end-1);
Xtrain = dummifyText(txtTran, vocab);

txtOutput = tAusten(2:end);
Ytrain = dummifyText(txtOutput, vocab);

load generateAustenNet.mat

startTxt = 'It is a tru';
startVar = dummifyText(startTxt, vocab);

[net, predicted] = predictAndUpdateState(net, startVar);

[~, idx] = max(predicted(:,end));
genChar = char(vocab(idx));
genTxt = [startTxt, genChar];

for n=1:200
    genvar = dummifyText(genChar, vocab);
    [net, predicted] = predictAndUpdateState(net, genvar);
    [~, idx] = max(predicted(:,end));
    genChar = char(vocab(idx));
    genTxt = [startTxt, genChar];
end