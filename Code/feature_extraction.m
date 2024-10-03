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

% Initialize storage for features and labels
features = [];%empty array forfeatures
labels = {};  %labels of audio signals(folder locations)

for i = 1:length(folders)
    currentFolder = folders{i}; 
    files = dir(fullfile(currentFolder, '*.wav')); %process through the audio files
    
    % Process files in chunks
    num_chunks = ceil(length(files) / chunk_size);
    
    for k = 1:num_chunks
        start_idx = (k-1) * chunk_size + 1;
        end_idx = min(k * chunk_size, length(files));
        chunk_files = files(start_idx:end_idx);

        %extract features from each audio file
        for j = 1:length(chunk_files)
            [signal, Fs] = audioread(fullfile(currentFolder, chunk_files(j).name));
            signal = single(signal); % Use single precision
            
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
            labels = [labels; {folders{i}}];  % Add corresponding label
        end
    end
end

% Convert labels to a cell array of strings
labels = string(labels);

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

data.Label = labels; % Add labels to the table

% Define path and write the table to the CSV file
csvFilePath = fullfile(folderPath, 'features.csv');
writetable(data, csvFilePath);

disp(['CSV file saved to: ',csvFilePath]);
