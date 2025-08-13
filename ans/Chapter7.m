% Import
I = imread('Madeline.jpg');
% imshow(I)

% Bounding box [x,y,w,h]
bbox = [788 367 953 959];
label = 'Madeline';

% Insert a shape + annotate
I = insertObjectAnnotation(I,'rectangle',bbox,label, ...
                           'LineWidth',20,'FontSize',72);
% imshow(I)

% Datastore
imgds = imageDatastore('datasets\petImages','IncludeSubfolders',true);

% Load in the labeled data created
load petGT.mat

% Create 5 anchor boxes
ab = anchorBoxMaker(petGT,5);

% Load in network for transfer learning
net = resnet18;

% Adjust the end of the network to add YOLO layers
net = yolov2Layers([224,224,3],5,ab,net,'res5b_relu', ...
                   'ReorgLayerSource','res3a_relu');

% Training options
opts = trainingOptions('adam','MiniBatchSize', 32, ...
                       'InitialLearnRate', 1e-3, ...
                       'MaxEpochs', 20, ...
                       'LearnRateSchedule', 'piecewise', ...
                       'LearnRateDropPeriod', 10);

% Train an object detector
% petdet = trainYOLOv2ObjectDetector(petGT,net,opts);
load petdet.mat

% Test the detector
I = imread('Ginny86.jpg');
% imshow(I)

[bboxes,scores,labels] = detect(petdet,I);
labels = cellstr(labels);
txt = strcat(labels,':',num2str(scores));

I = insertObjectAnnotation(I,'rectangle',bboxes,txt, ...
    'LineWidth',5,'FontSize',18);
imshow(I)