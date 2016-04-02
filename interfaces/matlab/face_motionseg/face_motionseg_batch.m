function face_motionseg_batch(varargin)
%FACE_MOTION_SEG_BATCH Summary of this function goes here
%   Detailed explanation goes here

%% Parse input arguments
p = inputParser;
addRequired(p, 'inDir', @ischar);
addRequired(p, 'outDir', @ischar);
addRequired(p, 'landmarks', @ischar);
addParameter(p, 'indices', [], @isvector);
addParameter(p, 'minWidth', 0, @isscalar);
addParameter(p, 'minHeight', 0, @isscalar);
addParameter(p, 'verbose', 0, @isscalar);
parse(p,varargin{:});
indices = p.Results.indices;

%% Create output directory structure
outputPath = fullfile(p.Results.outDir, 'output');
segmentationsPath = fullfile(p.Results.outDir, 'segmentations');
framesPath = fullfile(p.Results.outDir, 'frames');
summariesPath = fullfile(p.Results.outDir, 'summaries');
if(~exist(outputPath, 'dir'))
    mkdir(outputPath);
    mkdir(segmentationsPath);
    mkdir(framesPath);
    mkdir(summariesPath);
end

%% Parse input directory
filt = '.*(avi|mp4|mkv)';
fileDescs = dir(p.Results.inDir);
fileNames = {fileDescs(~cellfun(@isempty,regexpi({fileDescs.name},filt))).name};
if(isempty(indices))
    indices = 1:length(fileNames);
elseif(max(indices) > length(fileNames) || min(indices) < 1)
    error(['indices must be from 1 to ' num2str(length(fileNames))]);
end

%% For each video file
for i = indices   
    vidFile = fullfile(p.Results.inDir, fileNames{i});
    [vidPath,vidName, vidExt] = fileparts(vidFile);
    
    %% Check for existing summary
    dstSummaryPath = fullfile(summariesPath, [vidName '_summary.csv']);
    if(exist(dstSummaryPath, 'file') == 2)
        disp(['Skipping "', [vidName vidExt], '" because it was previously processed']);
        continue;
    end
    
    %% Check resolution
    vid = VideoReader(vidFile);
    if(vid.Width < p.Results.minWidth || vid.Height < p.Results.minHeight)
        disp(['Skipping "', [vidName vidExt], '" because of low resolution']);
        continue;
    end
    
    %% Process
    clear vid;
    vidOutDir = fullfile(outputPath, vidName);
    mkdir(vidOutDir);
    disp(['Processing "', [vidName vidExt], '"']);
    face_motionseg(vidFile, vidOutDir, p.Results.landmarks,...
        'verbose', p.Results.verbose);
    
    %% Read summary
    srcSummaryPath = fullfile(vidOutDir, 'summary.csv');
    if(exist(srcSummaryPath, 'file') == 2)
        S = csvread(srcSummaryPath, 1, 0);
        ids = S(:,1);
        scores = S(:,2);
    else
        warning(['There was an unexpected problem while processing "', [vidName vidExt], '"']);
        continue;
    end
    
    %% Find best segmentations
    [pks seg_ids] = findpeaks(scores,ids,'MinPeakDistance',10,...
        'MinPeakHeight',0.85,'SortStr','descend');
    
    %% Copy segmentations
    for seg_id = seg_ids(1:min(5,end))'
        srcSegFileName = ['seg_' num2str(seg_id, '%04d') '.png'];
        dstFrameFileName = [vidName '_seg_' num2str(seg_id, '%04d') '.png'];
        srcSegPath = fullfile(vidOutDir, srcSegFileName);
        dstFramePath = fullfile(segmentationsPath, dstFrameFileName);
        copyfile(srcSegPath, dstFramePath);
    end
    
    %% Copy frames
    for seg_id = seg_ids(1:min(5,end))'
        srcFrameFileName = ['frame_' num2str(seg_id, '%04d') '.png'];
        dstFrameFileName = [vidName '_frame_' num2str(seg_id, '%04d') '.png'];
        srcFramePath = fullfile(vidOutDir, srcFrameFileName);
        dstFramePath = fullfile(framesPath, dstFrameFileName);
        copyfile(srcFramePath, dstFramePath);
    end 
    
    %% Copy summary
    copyfile(srcSummaryPath, dstSummaryPath); 
end

