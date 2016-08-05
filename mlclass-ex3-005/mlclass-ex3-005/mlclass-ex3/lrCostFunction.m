function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
XTimesTheta = X*theta; % n by p+1, and p+1 by 1, so n by 1, each row is a y hat
hypothesisFunc = sigmoid(XTimesTheta); % each row is h_theta(x(i)) in note

%components in the cost function J of logistic regression, according to
%lec6 note
logHypothesis = log(hypothesisFunc);
log1_Hypothesis = log(1-hypothesisFunc);

J = -((y'*logHypothesis)+((1-y)'*log1_Hypothesis))/m + lambda/(2*m)*(theta(2:end))'*theta(2:end);
grad = ((hypothesisFunc-y)'*X)'./m + [0;theta(2:end)]*lambda/m;

% =============================================================

grad = grad(:);

end
