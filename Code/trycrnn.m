% Load the trained CRNN model
load('trainedCRNN.mat', 'net');

% Load test data 
testData = readmatrix('C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\validator set\validatorcrnn.csv');  % Replace with actual test data file path

% Extract features and labels for the new test data
XTestNew = testData(:, 1:end-1); % Features 
YTestNew = testData(:, end);     % Labels 

% Convert the labels to categories
YTestNew = categorical(YTestNew, [1, 2, 3]);

% Reshape the test data to match the input format expected by CRNN
numFeatures = size(XTestNew, 2);
numSamples = size(XTestNew, 1);

XTestNew = reshape(XTestNew', [numFeatures, 1, numSamples]);
XTestCell = cell(numSamples, 1);
for i = 1:numSamples
    XTestCell{i} = XTestNew(:, :, i);
end

% Make predictions on the test data
YPredNew = classify(net, XTestCell);

% Calculate the accuracy of the predictions
accuracy = sum(YPredNew == YTestNew) / numel(YTestNew) * 100;
disp(['Test Accuracy on new data: ', num2str(accuracy), '%']);

% Display actual vs predicted classes for test samples
disp('Actual vs Predicted Labels:');
for i = 1:numel(YTestNew)
    disp(['Sample ', num2str(i), ': Actual = ', char(YTestNew(i)), ', Predicted = ', char(YPredNew(i))]);
end
