function [Each_Image] = SP_Image_Generator(Each_Segment,Sampling_Rate)
Each_Image = zeros(5,5,4);
for l=1:size(Each_Segment,1)
    P = Band_Power_Cal(Each_Segment(l,:),Sampling_Rate);
    Features(:,l) = P ; 
end

for i=1:size(Features,1)
    Each_Image(1,2,i) = Features(i,1);
    Each_Image(2,2,i) = Features(i,2);
    Each_Image(3,2,i) = Features(i,3);
    Each_Image(4,2,i) = Features(i,4);
    Each_Image(5,2,i) = Features(i,5);
    Each_Image(2,1,i) = Features(i,6);
    Each_Image(3,1,i) = Features(i,7);
    Each_Image(4,1,i) = Features(i,8);
    Each_Image(2,3,i) = Features(i,9);
    Each_Image(1,4,i) = Features(i,10);
    Each_Image(2,4,i) = Features(i,11);
    Each_Image(3,4,i) = Features(i,12);
    Each_Image(4,4,i) = Features(i,13);
    Each_Image(5,4,i) = Features(i,14);
    Each_Image(2,5,i) = Features(i,15);
    Each_Image(3,5,i) = Features(i,16);
    Each_Image(4,5,i) = Features(i,17);
    Each_Image(3,3,i) = Features(i,18);
    Each_Image(4,3,i) = Features(i,19);
end





end

