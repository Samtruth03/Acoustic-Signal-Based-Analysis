% Load audio signal
[noisySignal, Fs] = audioread('C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\test_cut\engine1_good\whitenoise_low_0.wav');
numSamples = 1500;  % Adjust this value as needed
% Set bandpass filter parameters
lowFreq = 500;    % Lower bound frequency in Hz
highFreq = 1500;  % Upper bound frequency in Hz
order = 4;        % Filter order

% Design a bandpass filter
[b, a] = butter(order, [lowFreq, highFreq] / (Fs / 2), 'bandpass');

% Apply bandpass filter to the noisy signal
cleanedSignal = filter(b, a, noisySignal);

% Save cleaned signal to a new audio file
audiowrite('cleaned_signal.wav', cleanedSignal, Fs);

% Play the cleaned signal as audio
sound(cleanedSignal, Fs);

% Plotting signals in a 1x3 layout for comparison
figure;
subplot(2, 1, 1);
plot(noisySignal(1:numSamples));
title('Noisy Signal (Segment)');
xlabel('Sample');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(cleanedSignal(1:numSamples));
title('Cleaned Signal (Segment)');
xlabel('Sample');
ylabel('Amplitude');


% Display success message
disp('Data cleaning complete. Cleaned signal saved and played. Plots generated for comparison.');
