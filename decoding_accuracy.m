% Input:
%   A: ncells x ntrials_1 x nframes
%   B: ncells x ntrials_2 x nframes
% 
% Output:
%   p: Decoding accuracy

clear; clc; close all;

A = zeros(50,130,100);
B = ones(50,120,100);

binsize = 9;
nsamples = 8;
nframes = size(A,3)-binsize;

accu = nan(nsamples,nframes);

for sample = 1:nsamples
	for frame = 1:nframes; tic
        x = A(:,:,frame:frame+binsize);
        y = B(:,:,frame:frame+binsize);
        
		accu(sample,frame) = lstm_decoder(x,y,0); toc
	end
end

plot(mean(accu,1))
ylim([0,1])

