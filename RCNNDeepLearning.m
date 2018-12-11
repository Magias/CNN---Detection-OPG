
load('Rectagleteethgt.mat');
gTruth.LabelDefinitions
%%
zubgt = selectLabels(gTruth,'Tooth');

trainingData = objectDetectorTrainingData(zubgt);
summary(trainingData)
%%
trainingData(1:4,:)

%%
% Read one of the images.
I = imread(trainingData.imageFilename{10});

% Insert the ROI labels.
I = insertShape(I, 'Rectangle', trainingData.Tooth{10});

% Resize and display image.
I = imresize(I,3);
figure
imshow(I)


%% Create CNN
% Split data into a training and test set.
idx = floor(0.8 * height(trainingData));
teethTrain = trainingData(1:idx,:);
teethValidation = trainingData(idx:end,:);

% Create image input layer.
inputLayer = imageInputLayer([32 32 3]);

% Define the convolutional layer parameters.
filterSize = [3 3];
numFilters = 30;

% Create the middle layers.
middleLayers = [
                
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)   
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)  
    reluLayer() 
    maxPooling2dLayer(3, 'Stride',2)    
    
    ];

finalLayers = [
    
    % Add a fully connected layer with 64 output neurons. The output size
    % of this layer will be an array with a length of 64.
    fullyConnectedLayer(64)

    % Add a ReLU non-linearity.
    reluLayer()

    % Add the last fully connected layer. At this point, the network must
    % produce outputs that can be used to measure whether the input image
    % belongs to one of the object classes or background. This measurement
    % is made using the subsequent loss layers.
    fullyConnectedLayer(width(teethTrain))

    % Add the softmax loss layer and classification layer. 
    softmaxLayer()
    classificationLayer()
];
layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

% Options for step 1.
options = trainingOptions('sgdm', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

%%  
doTrainingAndEval = true;

if doTrainingAndEval
    % Set random seed to ensure example training reproducibility.
    rng(0);
    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    detector = trainFasterRCNNObjectDetector(teethTrain, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'NumRegionsToSample', [256 128 256 128], ...
        'BoxPyramidScale', 1.2);
else
    % Load pretrained detector for the example.
    detector = data.detector;
end

% Read a test image.
I = imread(teethValidation.imageFilename{1});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

