load('randomForestModel.mat')


testFolder = uigetdir();  % user input the folder containing test audio files
files = dir(fullfile(testFolder, '*.wav'));  
numSamples = 1500; 

% Filter parameters
lowFreq = 500;    % Lower bound frequency in Hz
highFreq = 1500;  % Upper bound frequency in Hz
order = 4;        % Filter order

% Iterate over each audio file in the test folder
for i = 1:length(files)
    % Load and clean the audio signal
    filePath = fullfile(files(i).folder, files(i).name);
    [noisySignal, Fs] = audioread(filePath);
    [b, a] = butter(order, [lowFreq, highFreq] / (Fs / 2), 'bandpass');
    cleanedSignal = filter(b, a, noisySignal);
    
    % Extract features from the cleaned signal
    rmsValue = rms(cleanedSignal);
    varianceValue = var(cleanedSignal);
    meanValue = mean(cleanedSignal);
    crestFactor = max(abs(cleanedSignal)) / rmsValue;
    energy = sum(cleanedSignal.^2);
    
    % Calculate dominant frequency
    N = length(cleanedSignal);  % Number of samples
    fftSignal = fft(cleanedSignal);  % Compute the FFT
    f = (0:N-1)*(Fs/N);  % Frequency vector
    magnitude = abs(fftSignal(1:N/2));  % Magnitude of the first half of the spectrum 
    [~, idx] = max(magnitude);  % Find index of the maximum magnitude
    dominantFrequency = f(idx);  % Corresponding dominant frequency
    
    % Skewness
    skewnessValue = skewness(cleanedSignal);
    
    % Kurtosis
    kurtosisValue = kurtosis(cleanedSignal);
    
    % Zero Crossing Rate
    zeroCrossings = sum(abs(diff(sign(cleanedSignal)))) / N;
    
    % Peak-to-Peak Amplitude
    peakToPeak = max(cleanedSignal) - min(cleanedSignal);
    
    % Signal Entropy
    entropyValue = -sum((abs(cleanedSignal) / sum(abs(cleanedSignal))).^2 .* log2(abs(cleanedSignal) / sum(abs(cleanedSignal))));
    
    % Spectral Flux
    spectralFlux = sum(diff(magnitude).^2);
    
    % Root Mean Square of Frequency Domain 
    rmsFrequencyDomain = rms(magnitude);
    
    % Combine all extracted features into a feature vector
    extractedFeatures = [rmsValue, varianceValue, meanValue, crestFactor, energy, dominantFrequency, ...
                         skewnessValue, kurtosisValue, zeroCrossings, peakToPeak, entropyValue, spectralFlux, rmsFrequencyDomain];

    % Classify the signal with the Random Forest model
    predictedLabel = predict(randomForestModel, extractedFeatures);

    % Display results
    fprintf('File: %s\nPredicted Label: %s\n', files(i).name, string(predictedLabel));
end

% Summarize results across all files in the test set
disp('Testing completed for all files in the selected folder.');