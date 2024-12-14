% Define input and output folders for each class
inputFolders = { ...
    'C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\train_cut\engine1_good', ...
    'C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\train_cut\engine2_broken', ...
    'C:\Users\sonar\Desktop\Assignments\sem7 project\Dataset\train_cut\engine3_heavyload' ...
};

outputFolders = { ...
    'C:\Users\sonar\Desktop\Assignments\sem7 project\Cleaned Data\Cleaned_Good', ...
    'C:\Users\sonar\Desktop\Assignments\sem7 project\Cleaned Data\Cleaned_Broken', ...
    'C:\Users\sonar\Desktop\Assignments\sem7 project\Cleaned Data\Cleaned_HeavyLoad' ...
};

% Set bandpass filter parameters
lowFreq = 500;    % Lower bound frequency in Hz
highFreq = 1500;  % Upper bound frequency in Hz
order = 4;        % Filter order

% Loop over each folder to clean and save signals
for i = 1:length(inputFolders)
    % Get list of audio files in the current input folder
    audioFiles = dir(fullfile(inputFolders{i}, '*.wav'));
    
    % Create the output folder if it doesn't exist
    if ~exist(outputFolders{i}, 'dir')
        mkdir(outputFolders{i});
    end
    
    % Process each audio file in the folder
    for j = 1:length(audioFiles)
        % Construct the full file path
        audioFilePath = fullfile(audioFiles(j).folder, audioFiles(j).name);
        
        % Load audio signal
        [noisySignal, Fs] = audioread(audioFilePath);
        
        % Design a bandpass filter
        [b, a] = butter(order, [lowFreq, highFreq] / (Fs / 2), 'bandpass');
        
        % Apply the bandpass filter to the noisy signal
        cleanedSignal = filter(b, a, noisySignal);
        
        % Construct the output filename
        outputFileName = sprintf('clean_%s%d.wav', lower(extractAfter(outputFolders{i}, 'Cleaned_')), j);
        outputFilePath = fullfile(outputFolders{i}, outputFileName);
        
        % Save the cleaned signal to the output folder
        audiowrite(outputFilePath, cleanedSignal, Fs);
    end
end

disp('Data cleaning and saving complete for all folders.');
