function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
temp_matrix = X*Theta'-Y;  %%n_movie*n_user dimension matrix
temp_matrix2 = (temp_matrix).^2;
J = sum(temp_matrix2(R == 1))/2 + sum((Theta(:)).^2)/2*lambda + sum((X(:)).^2)/2*lambda;

for i = 1:size(X_grad,1) %% i = 1 to n_movie
    Theta_temp = Theta(R(i,:) == 1,:); %%R(i,:)==1 selects some #'s in [1,n_user]
    X_grad(i,:) = (temp_matrix(i,R(i,:) == 1)*Theta_temp) + lambda*X(i,:);
end

for j = 1:size(Theta_grad,1) %% j = 1 to n_movie
    X_temp = X(R(:,j) == 1,:); %% R(:,j)==1 selects some #s in [1,n_movie]
    Theta_grad(j,:) = (temp_matrix(R(:,j) == 1,j))'*X_temp + lambda*Theta(j,:);
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
