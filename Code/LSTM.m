% Load the feature-extracted data from your CSV file
data = readtable("F:\\Aerogel Project\\features.csv");

% Separate features and labels
features = data{:, 1:end-1};  % Extract all feature columns
labels = data.Label;           % Extract the label column

% Convert labels to categorical for the classifier
labels = categorical(labels);

% Define the number of time steps and features based on data dimensions
totalFeatures = size(features, 2);

% Try to determine a valid numTimeSteps and numFeatures
possibleNumTimeSteps = 5:totalFeatures; % Range to find a valid divisor
numTimeSteps = 0;
numFeatures = 0;

for t = possibleNumTimeSteps
    if mod(totalFeatures, t) == 0
        numTimeSteps = t;
        numFeatures = totalFeatures / t;
        break;
    end
end

% Check if we found valid dimensions
if numTimeSteps == 0
    error('Unable to find a compatible number of time steps and features.');
end

% Reshape the features for LSTM input (samples x time steps x features)
features = reshape(features, [], numTimeSteps, numFeatures);

% Split data into training and testing sets (80% training, 20% testing)
cv = cvpartition(labels, 'HoldOut', 0.2); % 80-20 split
XTrain = features(training(cv), :, :);
YTrain = labels(training(cv));
XTest = features(test(cv), :, :);
YTest = labels(test(cv));

% Define the LSTM network architecture
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(50, 'OutputMode', 'last') % LSTM layer with 50 units
    fullyConnectedLayer(numel(categories(labels)))
    softmaxLayer
    classificationLayer
];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the LSTM model
lstmModel = trainNetwork(XTrain, YTrain, layers, options);

% Test the model on the test set
YPred = classify(lstmModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Validation accuracy of LSTM: ', num2str(accuracy * 100), '%']);

% Display a confusion matrix for performance insights
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for LSTM Validation');
