function accu = svm_decoder(A,B,shuffle,varargin)
% Binary SVM decoder
%
% function accu = svm_decoder(A,B,varargin)
%
% Inputs
% A - C x R1 matrix
% B - C x R2 matrix
%

A = mean(A,3);
B = mean(B,3);

Data   = vertcat(A',B');
Label  = categorical( vertcat( zeros(size(A,2),1), zeros(size(B,2),1) + 1) );
Trials = 1:size(Data,1);

Train_trials = randsample(Trials,ceil(0.9*numel(Trials)));
Test_trials = setdiff(Trials,Train_trials);

Train_data  = Data(Train_trials,:);
Train_label = Label(Train_trials);

Test_data   = Data(Test_trials,:);
Test_label  = Label(Test_trials);

if shuffle; Train_label = Train_label(randperm(length(Train_label))); end

modl = fitcsvm(Train_data,Train_label);
pred = predict(modl,Test_data);
accu = sum(Test_label == pred) ./ numel(Test_label);

end
