% Import an image + AlexNet
net = alexnet;
I = imread('Harper.jpg');
I = imresize(I,[227 227]);
% imshow(I)

% Look at the output (activations) of the 1st conv layer
AC1 = activations(net,I,'conv1');

% Visualize activations
AC1 = rescale(AC1);
% montage(AC1)
% showActivationsForChannel(I,AC1,39)

% Activations for the 1st ReLU layer
AR1 = activations(net,I,'relu1');
AR1 = rescale(AR1);
% montage(AR1)
% showActivationsForChannel(I,AR1,39)

% Activations for the 1st pool layer
AP1 = activations(net,I,'pool1');
AP1 = rescale(AP1);
% montage(AP1)
% showActivationsForChannel(I,AP1,39)

% Fifth conv layer (deep in)
AC5 = activations(net,I,'conv5');
AC5 = rescale(AC5);
% montage(AC5)
% showActivationsForChannel(I,AC5,210)

% What does a fully connected layer do?
FC = activations(net,I,'fc6'); % vectorizes features and combines them

% What does a dropout layer do?
DO = activations(net,I,'drop6'); % randomly cuts features to prevent "overfitting"