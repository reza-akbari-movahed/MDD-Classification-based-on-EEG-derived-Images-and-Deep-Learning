clc
clear all
close all
%% Loading dataset
Link= input("Please enter the direction of the dataset in MAT file format \n");
Link = strcat(Link, '\EEG_Dataset_Mod_Channels.mat'); 
load(Link);
%% EEG Signal Slicing
Idx = 1; 
Sampling_Rate = EEG_Dataset{1, 1}.srate ;
lag = ceil(10/60);   % lag
m = ceil(10/60);   % embedding dimension
w1 = ceil(100/60);  % window (Theiler correction for autocorrelation)
w2 = ceil(410/60);  % window (used to sharpen the time resolution of synchronization measure)
pref = 0.01;

for i=1:length(EEG_Dataset)
    Signal_Segments = Slicing_Function(EEG_Dataset{i,1},1);
    for j=1:size(Signal_Segments,2)
        Each_Segment = double(Signal_Segments{1,j});
        SP_Images(:,:,:,Idx) = SP_Image_Generator(Each_Segment,Sampling_Rate);
        FC_Images(:,:,Idx) = SL_Cal(Each_Segment,lag,m,w1,w2,pref);
        Labels(Idx,1) = EEG_Dataset{i,2};
        Idx = Idx + 1; 
    end
end
save('Extracted_Images.mat','FC_Images','SP_Images','Labels')