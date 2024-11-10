% Load the feature-extracted data from your CSV file
data = readtable("F:\\Aerogel Project\\features.csv");

% Separate features and labels
features = data{:, 1:end-1};  % Extract all feature columns
labels = data.Label;           % Extract the label column

% Convert labels to categorical for the classifier
labels = categorical(labels);

% Split data into training and testing sets (80% training, 20% testing)
cv = cvpartition(labels, 'HoldOut', 0.2); % 80-20 split
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv));

% Train the k-NN model (use 'NumNeighbors' parameter to set the k value)
k = 5; % Define the number of neighbors, adjust based on validation results
kNNModel = fitcknn(XTrain, YTrain, 'NumNeighbors', k, 'Standardize', true);

% Test the model on the test set
YPred = predict(kNNModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Validation accuracy of k-NN: ', num2str(accuracy * 100), '%']);

% Display a confusion matrix for performance insights
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for k-NN Validation');
