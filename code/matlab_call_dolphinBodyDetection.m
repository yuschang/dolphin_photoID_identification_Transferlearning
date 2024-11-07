
clc
clear all

% call python file from matlab

% Creating an instance of MyReceiver with a string
fastsam_checkpoint1 = 'E:\DeepLearning\InstanceDetection\implementation\instanceDetection_YOLOv8m_dolphinBody.pt';

% initiate the HED object in python code
% note: this should initiate only once. if there need reinitiate the object
% run following code first: segmentationObject.cleanup;
dolphinBodyDetectionObject1 = py.dolphinDetector.dolphinBodyDetector(fastsam_checkpoint1);
% segmentationObject.cleanup; % make suree clear all the net after the job is finished



%%

% boxList = func_cropROI(tmpImgPath, dolphinBoxfromPy_r , roiScaleUpFactor)
saveFolder = 'F:\JTA_PhotoID\dolphinBodyDetection_yolo';


% Collect the image file name list
processingImgFolder = 'F:\JTA_PhotoID\PD_Oh';
processingImgFolder_dir_top = dir(processingImgFolder); 
processingImgFolder_dir_top(1:2)= [];


for ftn = 7:10 
    
    tmp_processingImgFolder = fullfile(processingImgFolder_dir_top(ftn).folder, processingImgFolder_dir_top(ftn).name);
    processingImgFolder_dir = dir(tmp_processingImgFolder); 
    processingImgFolder_dir(1:2)= [];
    
    trainedImgSize = [1006, 708]; % the image size of trained image 
    
    close all
    for fn = 1: length(processingImgFolder_dir)
        disp(['Processing: ' num2str(fn/length(processingImgFolder_dir)*100) ' %' ])
        
        %** detect the dolphin
        tmpImgPath = fullfile(processingImgFolder_dir(fn).folder, processingImgFolder_dir(fn).name);
        [~, ~, formats] = fileparts(tmpImgPath);
        if strcmp(formats,'.jpg') || strcmp(formats,'.JPG')

            boxList_yolo = dolphinBodyDetectionObject1.detect(tmpImgPath, uint16(trainedImgSize));
            
            %** get the raw dolphin body rectangular box
            [dolphinBoxfromPy_r, dolphinBoxfromPy] = func_getBoxList(boxList_yolo, trainedImgSize); % relative box coordi, org pixel coordi
            % box coordinate format: (left, top, right, bottom)
        
            %** crop the square dolphin body ROI and save it to images
            if ~isempty(dolphinBoxfromPy_r)
                roiScaleUpFactor = 1.1;
                boxList = func_cropROI(tmpImgPath, dolphinBoxfromPy_r , roiScaleUpFactor, saveFolder);
            end

        end
    
    end

end