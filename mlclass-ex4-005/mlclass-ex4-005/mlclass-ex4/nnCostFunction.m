function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); %% 25 * 401
 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));   %% 10 * 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
codedY = zeros(m,num_labels); 
%%each row is coded y from the original y for one observation
%%e.g. if codedY(2,:)=[0,0,0,0,1,0,0,0,0,0], it represents that y(2,:)=5
for i = 1:m
    codedY(i,y(i,1)) = 1;
end;
%%add a bias 1 to each observation
X = [ones(m,1),X];  %%5000 * 401

%%calculate a's in layer 2 
z2 = X*Theta1';     %%5000 * 25
a2 = sigmoid(z2);   %%5000 * 25

%%add a bias 1 to each observation
a2 = [ones(m,1),a2];  %% 5000 * 26

%%calculate a's in layer 3, i.e. the output layer
z3 = a2*Theta2';  %% 5000*10
a3 = sigmoid(z3); %% 5000*10

%%calculate the cost
logOutput = log(a3);
log1_Output = log(1-a3);
coded1_Y = 1 - codedY;
temp = codedY.*logOutput + coded1_Y.*log1_Output;
J = - sum(temp(:)) / m;

%%add the regularization term
%%exclude the parameters for bias terms in thetas first
Theta1New = Theta1(:,2:end);  %%25 * 400
Theta2New = Theta2(:,2:end);  %%10 * 25
temp1 = Theta1New.^2;
temp2 = Theta2New.^2;
regularTerm = (sum(temp1(:)) + sum(temp2(:)))*lambda/(2*m);

J = J + regularTerm;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
a1 = X; %% 5000 * 401
error3 = a3 - codedY; %%5000 * 10
error2 = zeros(m,hidden_layer_size);  %%5000 * 25
delta1 = zeros(size(Theta1));  %%25 * 401
delta2 = zeros(size(Theta2));  %%10 * 26
for i = 1:m
                  % 1*10           10*25              1*25
    error2(i,:) = error3(i,:) * Theta2New .* sigmoidGradient(z2(i,:));
    delta1 = delta1 + (error2(i,:))' * a1(i,:);
    delta2 = delta2 + (error3(i,:))' * a2(i,:);
end;
Theta1_grad = delta1 / m;
Theta2_grad = delta2 / m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regulTerm_Theta1 = [zeros(hidden_layer_size,1) Theta1(:,2:end)]*lambda/m;
regulTerm_Theta2 = [zeros(num_labels,1) Theta2(:,2:end)]*lambda/m;
Theta1_grad = Theta1_grad + regulTerm_Theta1;
Theta2_grad = Theta2_grad + regulTerm_Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
