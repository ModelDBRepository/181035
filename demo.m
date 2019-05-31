%%% Example code for FFV1MT code
%%% v0.1
%%% 26/03/2015
%%%
%%% M.Chessa and F. Solari
%%% University of Genoa, ITALY
%%%
%%% manuela.chessa@unige.it
%%% fabio.solari@unige.it
%%%
%%% REF PAPER:
%%% F. Solari, M. Chessa, N. Medathati, and P. Kornprobst. 
%%% What can we expect from a V1-MT feedforward architecture for optical flow estimation? 
%%% Submitted to Signal Processing: Image Communication, 2015.
%%%

clear all

n_scales =6;    %pyramidal scales
th=1e-4;        %motion energy threshold
th2=1e-3;       %motion energy threshold after spatial pooling
n_filters=12;   %spatial orientations for V1 level
vel=[-0.9 -0.6 -0.4 0 0.4 0.6 0.9];     %component velocities for V1 level
D=2;            %speed directions for MT level

load yosemite
O =FFV1MT(I(:,:,2:6),n_scales,th,th2,vel,n_filters,D);

%%%%%% Visualization of the results
figure, imagesc(O(:,:,1))
title('Vx');

figure, imagesc(O(:,:,2))
title('Vy');

%%%%%% Visualization of the results (with Middlebury code available from http://vision.middlebury.edu/flow/code/flow-code-matlab.zip ) 
%img=flowToColor(O);
%figure, image(img)
