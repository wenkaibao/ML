function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
z2 = X*Theta1';  %5000 X 25 matrix, each row is a z2 for a x input
a2 = sigmoid(z2); % elementwise apply the logistic function to each z2
a2 = [ones(m,1) a2]; %add bias/intercept, so 5000 X 26 now

z3 = a2*Theta2'; %5000 X 10, each row is a z3 for a a2 input
a3 = sigmoid(z3);
[maxProb, maxProbLoc] = max(a3, [], 2);
p = maxProbLoc(:);

% =========================================================================


end
