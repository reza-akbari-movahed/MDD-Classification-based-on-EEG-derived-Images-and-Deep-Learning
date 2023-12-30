# MDD-Classification-based-on-EEG-derived-Images-and-Deep-Learning
## Introduction
This is the official implementation of "Automated major depressive disorder diagnosis using a dual-input deep learning model and image generation from EEG signals" paper. The present paper introduces a deep learning approach based on image construction from EEGs. Two images are constructed from EEGs based on spectral and functional connectivity features. Afterward, the constructed images are applied to a two-stream convolutional neural network, and the outputs are concatenated. Finally, the concatenating result is applied to a sequential model of long–short-term memory, fully connected, and softmax layers to classify each sample into the MDD and healthy control (HC) classes. To validate the proposed approach, a public EEG dataset was used consisting of EEG data recorded from 34 MDD patients and 30 HC-matched participants. This framework obtained an AC of 98.03%, SE of 98.85%, SP of 97.19%, F1 of 98.07%, and FDR of 2.69% for the random splitting assessment method and achieved an average AC of 99.11%, SE of 98.97%, SP of 99.25%, F1 of 99.13%, and FDR of 0.71% using a 10-fold cross-validation process. Considering the accurate performance of the proposed method, it can be developed as a computer-aided diagnosis tool to diagnose MDD automatically. More details of the paper are provided in the below link. <br />
https://doi.org/10.1080/17455030.2023.2187237
## Dependencies
The codes for image generation from EEG signals are implemented in MATLAB 2019b, and the deep learning part of the framework is implemented in Python using the Tensorflow/Keras package. 
## Dataset Link 
The preprocessed EEG signals are saved into a MAT file and can be downloaded using the below link. The label 1 refers to HC samples and the label -1 refers to MDD cases. <br />
https://drive.google.com/file/d/1A3Xyon397om1t5WxINzZzGMBjiaRd0d0/view?usp=sharing
## How to run 
1. Clone the GitHub repository. 
2. Download the dataset using the provided link and copy it to the main directory of the project.
3. Go to the Image Generation Folder and run Data_Generator.m.
   - When running, you should enter the direction which contains the MAT-file format dataset. For example, if the file is located in F:\\myfolder, you should enter "F:\\myfolder".
   - Note that the MAT-file save format should be set to -v7.3 in the settings of MATLAB (You can set this using the preferences menu in MATLAB).
   - It results in creating a file named Extracted_Images.mat. Copy and paste this file into the Training and Evaluating folder 
4. Go to the Training and Evaluating folder and run one of the following codes in Python based on the evaluation method.
   - Final_Code_Cross_Validation_With_Val.py
   - Final_Code_Cross_Validation_Without_Val.py
   - Final_Code_Random_Splitting.py
## Python Requirments 
The training and evaluating part of the proposed framework is implemented on Python 3.7. The required packages and their versions for running the codes in Python are listed in Python_requirments.txt file.
