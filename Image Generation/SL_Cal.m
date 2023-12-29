function [Matrix_Results] = SL_Cal(EEG_Signal,lag,m,w1,w2,pref)
% This code extractes the synchronization likelihood features
% EEG_Signal = EEG_Data.data;
EEG_Signal = EEG_Signal';
Matrix_Results = H_sl(EEG_Signal,lag,m,w1,w2,pref);

end

