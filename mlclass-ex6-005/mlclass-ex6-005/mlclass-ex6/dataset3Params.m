function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_candidates = [.01,.03,.1,.3,1,3,10,30];
sigma_candidates = C_candidates;
C_sigma = setprod(C_candidates,sigma_candidates);
C_sigma_error = [C_sigma, zeros(size(C_sigma,1),1)];

C = 1;
sigma = 1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i = 1:size(C_sigma_error,1)
    model = svmTrain(X, y, C_sigma_error(i,1), @(x1, x2) gaussianKernel(x1, x2, C_sigma_error(i,2)));
    predictions = svmPredict(model, Xval);
    C_sigma_error(i,3) = mean(double(predictions ~= yval));
end

[min_error,min_error_row] = min(C_sigma_error(:,3));
C = C_sigma_error(min_error_row,1);
sigma = C_sigma_error(min_error_row,2);

% =========================================================================

end
