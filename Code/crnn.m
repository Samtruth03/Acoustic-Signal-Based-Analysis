% Load data from your feature CSV file
data = readmatrix('crnnfeatures.csv');
X = data(:, 1:end-1); % Features
Y = data(:, end);     % Labels 

% Convert labels to categories with explicit categories
Y = categorical(Y, [1 2 3]);


% Check for undefined labels 
if any(isundefined(Y))
    disp('Warning: Undefined labels detected in Y!');
    % Reorder the categories to make sure they are valid
    Y = reordercats(Y, [1, 2, 3]);
end


% Check that the labels only contain valid values 
validLabels = [1, 2, 3];  % Define valid categories

if any(~ismember(double(Y), validLabels))
    disp('Invalid labels detected in Y!');
    disp('Invalid labels in Y:');
    disp(Y(~ismember(double(Y), validLabels)));  % Show the invalid labels
    error('Labels in Y contain invalid values!');
end

% Partition data into training and testing sets
cv = cvpartition(numel(Y), 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);

% Get the training and testing labels
YTrain = Y(idxTrain);
YTest = Y(idxTest);



% Check that YTrain and YTest only contain valid values 
if any(~ismember(double(YTrain), validLabels)) || any(~ismember(double(YTest), validLabels))
    error('Labels in YTrain or YTest contain invalid values!');
end

% Get dimensions
numFeatures = size(X, 2);
numSamples = size(X, 1);

% Reshape data and convert to cell array for sequence input
X = reshape(X', [numFeatures, 1, numSamples]);
XCell = cell(numSamples, 1);
for i = 1:numSamples
    XCell{i} = X(:, :, i);
end

XTrain = XCell(idxTrain);
XTest = XCell(idxTest);

% Define the CRNN architecture
layers = [
    sequenceInputLayer([13, 1])  % Input layer with feature dimension
    flattenLayer  % convert input to scalar size for LSTM
    lstmLayer(64, 'OutputMode', 'last')
    dropoutLayer(0.3)
    fullyConnectedLayer(3)  % 3 classes
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict labels on the test set
YPred = classify(net, XTest);

% Calculate the accuracy
accuracy = sum(YPred == YTest) / numel(YTest) * 100;
disp(['Test Accuracy: ', num2str(accuracy), '%']);

% Display actual vs predicted classes for test samples
disp('Actual vs Predicted Labels:');
for i = 1:numel(YTest)
    disp(['Sample ', num2str(i), ': Actual = ', char(YTest(i)), ', Predicted = ', char(YPred(i))]);
end
save('trainedCRNN.mat', 'net');