net = alexnet;
I = imread("harper.jpg");
I = imresize(I, [227 227]);
%imshow(I)


% check output of activcation of 1st layer
ac1 = activations(net, I, 'conv1');


%visualization
ac1 = rescale(ac1)
%montage(ac1)
%showActivationsForChannel(I, ac1, 39)

%relu act
ar1 = activations(net, I, 'relu1');
ar1 = rescale(ar1)
%montage(ac1)
showActivationsForChannel(I, ar1, 39)


%pool layer act
ap1 = activations(net, I, 'pool1');
ap1 = rescale(ap1)
%montage(ap1)
showActivationsForChannel(I, ap1, 39)