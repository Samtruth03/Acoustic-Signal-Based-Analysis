% Load the feature-extracted data from your CSV file
data = readtable('C:\Users\sonar\Desktop\Assignments\sem7 project\Code\features.csv');
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

% Train the Random Forest model with 50 trees (adjust as needed)
randomForestModel = fitcensemble(XTrain, YTrain, 'Method', 'Bag', 'NumLearningCycles', 50);

% Test the model on the test set
YPred = predict(randomForestModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Validation accuracy of Random Forest: ', num2str(accuracy * 100), '%']);

% Display a confusion matrix for performance insights
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for Random Forest Validation');
