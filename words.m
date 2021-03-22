%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 19;  % 
hidden_layer_size = 25;   % 25 hidden units
num_labels = 1497;          % 1497 labels, one for each word
training_limit = 10000;   %Over 70,000 examples was too much, 

% Load Training Data
printf('Select data files:\n')

y = [];
ytest = [];
finalX = [];
finaltest = [];
    
filename = uigetfile();
    
disp(filename);
load(filename);
m = size(X, 1);
if m > training_limit
    X = X(randperm(m), :);  % Shuffle the rows so that a different set is used each time
    X = X(1:training_limit, :);
    m = size(X, 1);
end

X = X(randperm(m), :);  % Shuffle the rows so that a different set is used for the test set each time

y = X(:,end:end); % The expected outcome is the last element in the list
X = X(:, 1:end-1); % Cut it off from the rest of the data
        
    
printf('Examples in training set: ');
printf(num2str(m));
printf('\n');
pause

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 10000);

%  used for feature regularization
lambda = 10000;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

save("-v7", "Theta_output.mat", "Theta1", "Theta2");
