function face_motionseg(varargin)
%FACE_MOTION_SEG Summary of this function goes here
%   Detailed explanation goes here

%% Parse input arguments
p = inputParser;
addRequired(p, 'videoFile', @ischar);
addRequired(p, 'outDir', @ischar);
addRequired(p, 'landmarks', @ischar);
addParameter(p, 'verbose', 0, @isscalar);
parse(p,varargin{:});

%% Execute face motion segmentation
exeName = 'face_motionseg';
[status, cmdout] = system([exeName ' "' p.Results.videoFile...
    '" -o "' p.Results.outDir '" -l "' p.Results.landmarks...
    '" -v ' num2str(p.Results.verbose)]);
if(status ~= 0)
    error(cmdout);
end

end

