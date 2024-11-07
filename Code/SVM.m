% Load feature-extracted data from your CSV file
data = readtable('C:\Users\sonar\Desktop\Assignments\sem7 project\Code\features.csv');

% Separate features and labels
features = data{:, 1:end-1};  % All columns except the last (features)
labels = data.Label;           % Label column (last column assumed)

% Convert labels to categorical (for classification)
labels = categorical(labels);

% Standardize the features (mean=0, variance=1)
features = normalize(features);

% Split data into training and testing sets (80% training, 20% testing)
cv = cvpartition(labels, 'HoldOut', 0.2); % 80-20 split
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv));

% Train the multi-class SVM model using fitcecoc
SVMModel = fitcecoc(XTrain, YTrain, 'Coding', 'onevsall');  % Removed 'Standardize' here

% Test the model on the test set
YPred = predict(SVMModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Validation accuracy of SVM: ', num2str(accuracy * 100), '%']);

% Display a confusion matrix to analyze performance
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for Multi-Class SVM Validation');
