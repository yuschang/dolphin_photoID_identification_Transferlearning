
% demo to call the null class rejection

clc
clear all

% call python file from matlab

% Creating an instance of MyReceiver with a string
model_checkpoint = 'D:\[IndivisualClassification]\Development\KeyPoint_detection\colab_roboflow_keypoint_detection_8points\implementation\keypointDetection_yolov8m_pos.pt';

% note: this should initiate only once. if there need reinitiate the object
% run following code first: segmentationObject.cleanup;
keyPointDetectorObj = py.keyPointDetector.keyPointDetector(model_checkpoint);
% segmentationObject.cleanup; % make suree clear all the net after the job
% is finished % need to do it later


% 109 31

%%

testimgfolder = 'D:\[IndivisualClassification]\Development\Individual dolphin\084';


imgList = func_getFileNameList_Light(testimgfolder, 'jpg');


%%

fn = fix(rand(1)*length(imgList)+1);

keypoints_in_pixel = keyPointDetectorObj.detect(fullfile(testimgfolder, imgList{fn}));
keypoints_matlab_matrix = double(keypoints_in_pixel);

close all


figure
imagesc(imread(fullfile(testimgfolder, imgList{fn}))); hold on
for roiN = 1: size(keypoints_matlab_matrix,1)
    tmpROIKeyPoint(:,1) = [keypoints_matlab_matrix(roiN,:,1)]';
    tmpROIKeyPoint(:,2) = [keypoints_matlab_matrix(roiN,:,2)]';
    scatter(tmpROIKeyPoint(:,1), tmpROIKeyPoint(:,2), 20, 'go' ,'filled'); hold on
    scatter(tmpROIKeyPoint(4,1), tmpROIKeyPoint(4,2), 20, 'ro' ,'filled'); hold on
end
hold off
axis image