%% Parse directories
descs = dir;
descs = descs(3:end);
descs = descs([descs.isdir]);
dirNames = {descs.name};

% Add all subfolders to search path
scriptPath = mfilename('fullpath');
[rootPath, filename, fileextension]= fileparts(scriptPath);
for i = 1:length(dirNames)
    addpath(fullfile(rootPath, dirNames{i}));
end