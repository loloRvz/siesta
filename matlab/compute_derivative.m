clc;
clear all;
close all;

% Get latest created file
d = dir('../data/*.csv');
[~, index]   = max([d.datenum]);
dataset = readmatrix(fullfile(d(index).folder, d(index).name)); 
% dataset = readmatrix('../data/23-02-28--13-24-40_L1-step.csv');

% Isolate important data
t = dataset(:,1);
setpt = dataset(:,2);
pos = dataset(:,3);
current = dataset(:,5);


%% DERIVATIVE FILTER %%
Fs = 400;
Nf = 10; 
Fpass = 10; 
Fstop = 100;

d = designfilt('differentiatorfir', 'FilterOrder',Nf, ...
    'PassbandFrequency',Fpass, ...
    'StopbandFrequency',Fstop, ...
    'SampleRate',Fs);


%% COMPUTE DERIVATIVES %%
dt = 1/Fs;
delay = mean(grpdelay(d));
tt = t(1:end-delay);
tt(1:delay) = [];
ttt = tt(1:end-delay);
ttt(1:delay) = [];

% Velocity
vel = filter(d,pos)/dt;
% Delay
vel_d = vel;
vel_d(1:delay) = [];
vel_d(1:delay) = [];

% Acceleration
acc = filter(d,vel_d)/dt;
% Delay
acc_d = acc;
acc_d(1:delay) = [];
acc_d(1:delay) = [];


%% PLOT %%
figure(1)
hold on
plot(t,setpt - pi)
plot(t,pos - pi)
plot(tt,vel_d/10)
plot(ttt,acc_d/1000)
plot(t,current/1000)
yline(0,'k')
xlabel('Time [ms]')
ylabel('Amplitude []')
legend('Setpoint [rad]','Position [rad]','Velocity [10rad/s]','Acceleration[1000rad/s]','Current [A]')