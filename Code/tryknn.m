% Load the trained k-NN model
load("knn.mat");  % Ensure the trained model is saved as 'kNNModel.mat'

% Load the CSV of pre-extracted test features
testData = readtable("C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\validator set\validatorfeatures.csv");  % Path to your test feature CSV

% Separate features and actual labels 
features = testData{:, 1:end-1};  
actualLabels = testData.Label;     

% Convert actual labels to categories
actualLabels = categorical(actualLabels);

% Predict class for each row in the test feature CSV
YPred = predict(kNNModel, features);

% Display the predictions
disp('Predicted Classes for Test Signals:');
for i = 1:length(YPred)
    disp(['Signal ', num2str(i), ': Predicted Class - ', char(YPred(i))]);
    if ~isempty(actualLabels)
        disp(['Actual Class - ', char(actualLabels(i))]);
    end
end