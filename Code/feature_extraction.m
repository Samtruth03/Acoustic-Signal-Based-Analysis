% Define paths to your folders
folders = {'C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\train_cut\engine1_good', ...
           'C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\train_cut\engine2_broken', ...
           'C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\train_cut\engine3_heavyload'};
folderPath = 'C:\Users\sonar\Desktop\Assignments\sem7 project\Code'; 

% Define sampling frequency
Fs = 44100;

% Define FFT window length and chunk size
fft_window_length = 1024; % Set FFT window length as appropriate
chunk_size = 10; % Adjust based on memory constraints

% Noise Filtering Parameters
lowCutoff = 100;   % Lower frequency limit (Hz)
highCutoff = 8000; % Upper frequency limit (Hz)

% Initialize storage for features and labels
features = [];
labels = {};

% Signal Cleaning Function
function [cleaned_signals, cleaned_filenames, cleaned_folders] = clean_all_signals(folders, Fs, lowCutoff, highCutoff)
    cleaned_signals = {};
    cleaned_filenames = {};
    cleaned_folders = {};
    
    % Iterate through all folders
    for i = 1:length(folders)
        currentFolder = folders{i};
        files = dir(fullfile(currentFolder, '*.wav'));
        
        % Process each file in the folder
        for j = 1:length(files)
            % Read audio file
            [signal, Fs] = audioread(fullfile(currentFolder, files(j).name));
            signal = single(signal); % Use single precision
            
            % Detailed diagnostic print
            fprintf('File: %s\n', files(j).name);
            fprintf('Original Signal - Length: %d, Min: %f, Max: %f, Mean: %f, Std: %f\n', ...
                length(signal), min(signal), max(signal), mean(signal), std(signal));
            
            try
                % Design Butterworth bandpass filter
                [b, a] = butter(6, [lowCutoff highCutoff]/(Fs/2), 'bandpass');
                
                % Apply zero-phase filtering
                cleaned_signal = filtfilt(b, a, signal);
                
                % Apply Hamming window for edge tapering
                window = hamming(length(cleaned_signal));
                cleaned_signal = cleaned_signal .* window;
                
                % Diagnostic print for cleaned signal
                fprintf('Cleaned Signal - Length: %d, Min: %f, Max: %f, Mean: %f, Std: %f\n', ...
                    length(cleaned_signal), min(cleaned_signal), max(cleaned_signal), ...
                    mean(cleaned_signal), std(cleaned_signal));
                
                % Store cleaned signal and its filename
                cleaned_signals{end+1} = cleaned_signal;
                cleaned_filenames{end+1} = files(j).name;
                cleaned_folders{end+1} = currentFolder;
                
            catch ME
                warning('Error processing %s: %s', files(j).name, ME.message);
            end
        end
    end
end

% Clean all signals first
[cleaned_signals, cleaned_filenames, cleaned_folders] = clean_all_signals(folders, Fs, lowCutoff, highCutoff);

% Process cleaned signals
for i = 1:length(cleaned_signals)
    signal = cleaned_signals{i};
    
    % fit signal to fft window length
    if length(signal) < fft_window_length
        % Pad signal if shorter than FFT window length
        signal = [signal; zeros(fft_window_length - length(signal), 1)];
    elseif length(signal) > fft_window_length
        % Truncate signal if longer than FFT window length
        signal = signal(1:fft_window_length);
    end
    
    % Apply FFT
    Y = abs(fft(signal, fft_window_length));  % FFT magnitude
    N = length(Y);
    f = (0:N-1) * (Fs/N);  % Frequency vector

    % Feature extraction
    rms_value = rms(signal); %rms calculation
    mean_value = mean(signal);%mean calculation
    variance_value = var(signal);%variance calculation
    crest_factor = max(abs(signal)) / rms_value; %crest factor calculation
    energy = sum(signal.^2); %energy calculation

    % Calculate dominant frequency
    [~, idx] = max(Y);
    dominant_freq = f(idx);  % Dominant frequency based on FFT peak
    
    % store extracted features into "features" array
    current_features = [rms_value, mean_value, variance_value, crest_factor, energy, dominant_freq];
    features = [features; current_features];  % Add to feature row
    labels = [labels; cleaned_folders{i}];  % Add corresponding label
end

% Check dimensions of features and labels
disp(['Size of features: ', num2str(size(features))]);
disp(['Size of labels: ', num2str(size(labels))]);

% Correct variable names to match the number of features
variableNames = {'RMS', 'Mean', 'Variance', 'CrestFactor', 'Energy', 'DominantFreq'};
if size(features, 2) == length(variableNames)
    data = array2table(features, 'VariableNames', variableNames);
else
    % Handle the case where feature count does not match variable names
    error('Mismatch between the number of features and the number of variable names.');
end

% Add labels to the table
data.Label = labels;

% Define path and write the table to the CSV file
csvFilePath = fullfile(folderPath, 'features.csv');
writetable(data, csvFilePath);

disp(['CSV file saved to: ',csvFilePath]);