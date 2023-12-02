function accu = lstm_decoder(A,B,shuffle,varargin)
% LSTM decoder
%
% function accu = lstm_decoder(X,Y,varargin)
%
% Inputs (required)
% X - C x T x R1 matrix
% Y - C x T x R2 matrix
%

A = permute(A,[1,3,2]);
B = permute(B,[1,3,2]);

Data   = cat(3,A,B); 
Label  = categorical( vertcat( zeros(size(A,3),1), zeros(size(B,3),1) + 1) );

Trials = 1:size(Data,3);

Train_trials = randsample(Trials,ceil(0.9*numel(Trials)));
Test_trials = setdiff(Trials,Train_trials);

for r = 1:numel(Train_trials); Train_data{r} = Data(:,:,Train_trials(r)); end %#ok<*AGROW>
for r = 1:numel(Test_trials);  Test_data{r}  = Data(:,:,Test_trials(r)); end

Train_label = Label(Train_trials);
Test_label  = Label(Test_trials);

if shuffle; Train_label = Train_label(randperm(length(Train_label))); end

inputSize = size(A,1);
numHiddenUnits = 128;
numClasses = numel(unique(Label));

layers = [ ...
	sequenceInputLayer(inputSize)
	lstmLayer(numHiddenUnits,'OutputMode','last')
	fullyConnectedLayer(numClasses)
	softmaxLayer
	classificationLayer];

options = trainingOptions('adam', ...
	'ExecutionEnvironment','cpu', ...
	'WorkerLoad',1, ...
	'GradientThreshold',1, ...
	'MaxEpochs',100, ...
	'MiniBatchSize',4, ...
	'SequenceLength','longest', ...
	'Shuffle','never', ...
	'Verbose',0, ...
	'Plots','none');

net = trainNetwork(Train_data,Train_label,layers,options);

LPred = classify(net,Test_data,'MiniBatchSize',27, 'SequenceLength','longest');

accu = sum(LPred == Test_label) ./ numel(Test_label);

end
