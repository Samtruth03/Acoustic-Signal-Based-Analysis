% Load the feature-extracted data from your CSV file
data = readtable('C:\Users\sonar\Desktop\Assignments\sem7 project\Code\features.csv');
features = data{:, 1:end-1};  % Extract all feature columns 
labels = data.Label;          % Extract the label column

% Convert labels to categorical for classification
labels = categorical(labels);

% Split the data into training and testing sets 
cv = cvpartition(labels, 'HoldOut', 0.2);  % 80-20 split
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv));

% Train the Random Forest model 
randomForestModel = fitcensemble(XTrain, YTrain, 'Method', 'Bag', 'NumLearningCycles', 50);



% Test the model on the test set
YPred = predict(randomForestModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Random Forest model accuracy: ', num2str(accuracy * 100), '%']);


save('randomForestModel.mat', 'randomForestModel');
