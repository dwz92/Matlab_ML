I = imread("madeline.jpg");



bbox = [788 367 953 959];
label = "Madeline";

I = insertObjectAnnotation(I, 'rectangle', bbox, label, ...
    LineWidth=20, ...
    FontSize=72);


imgds = imageDatastore('datasets\petImages\', 'IncludeSubfolders',true);

load petGT.mat

ab = anchorBoxMaker(petGT, 5);


net = resnet18;

net = yolov2Layers([224 224 3], 5, ab, net, 'res5b_relu', ...
    'ReorgLayerSource', 'res3a_relu');

opts = trainingOptions('adam', 'MiniBatchSize',32, ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=20,...
    LearnRateSchedule='piecewise',...
    LearnRateDropPeriod=10);

%petdet = trainYOLOv2ObjectDetector(petGT, net, opts);

load petdet.mat

I = imread("Ginny86.jpg")

[bbox, scores, labels] = detect(petdet, I);

labels = cellstr(labels);
txt = strcat(labels, ':', num2str(scores));

I = insertObjectAnnotation(I, "rectangle", bbox, txt,LineWidth=5, FontSize=8);
imshow(I)