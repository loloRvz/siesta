clc;
clear all;
close all;

dataset = readmatrix('../data/23-02-28--13-24-40_L1-step.csv');
times = dataset(:,1)
positions = dataset(:,3)


Fs = 400;
Nf = 50; 
Fpass = 100; 
Fstop = 120;

d = designfilt('differentiatorfir','FilterOrder',Nf, ...
                                   'PassbandFrequency',Fpass, ...
                                   'StopbandFrequency',Fstop, ...
                                   'SampleRate',Fs);






