function net(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size)
  model = init_model(n_hid);
  from_data_file = load('data.mat');
  datas = from_data_file.data;
  n_training_cases = size(datas.training.inputs, 2);
  if n_iters ~= 0,
    fprintf('Now testing the gradient on the whole training set... ');
    test_gradient(model, datas.training, wd_coefficient);
  end

  % optimization
  theta = model_to_theta(model);
  momentum_speed = theta * 0;
  training_data_losses = [];
  validation_data_losses = [];
  if do_early_stopping,
    best_so_far.theta = -1; % this will be overwritten soon
    best_so_far.validation_loss = inf;
    best_so_far.after_n_iters = -1;
  end
  for optimization_iteration_i = 1:n_iters,
    model = theta_to_model(theta);
    
    training_batch_start = mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1;
    training_batch.inputs = datas.training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    training_batch.targets = datas.training.targets(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    gradient = model_to_theta(grad(model, training_batch, wd_coefficient));
    momentum_speed = momentum_speed * momentum_multiplier - gradient;
    theta = theta + momentum_speed * learning_rate;

    model = theta_to_model(theta);
    training_data_losses = [training_data_losses, cost(model, datas.training, wd_coefficient)];
    validation_data_losses = [validation_data_losses, cost(model, datas.validation, wd_coefficient)];
    if do_early_stopping && validation_data_losses(end) < best_so_far.validation_loss,
      best_so_far.theta = theta; % this will be overwritten soon
      best_so_far.validation_loss = validation_data_losses(end);
      best_so_far.after_n_iters = optimization_iteration_i;
    end
    if mod(optimization_iteration_i, round(n_iters/10)) == 0,
      fprintf('After %d optimization iterations, training data cost is %f, and validation data cost is %f\n', optimization_iteration_i, training_data_losses(end), validation_data_losses(end));
    end
    if optimization_iteration_i == n_iters, % check gradient again, this time with more typical parameters and with a different data size
      fprintf('Now testing the gradient on just a mini-batch instead of the whole training set... ');
      test_gradient(model, training_batch, wd_coefficient);
    end 
  end
  if do_early_stopping,
    fprintf('Early stopping: validation cost was lowest after %d iterations. We chose the model that we had then.\n', best_so_far.after_n_iters);
    theta = best_so_far.theta;
  end
  % the optimization is finished. Now do some reporting.
  model = theta_to_model(theta);
  if n_iters ~= 0,
    clf;
    hold on;
    plot(training_data_losses, 'b');
    plot(validation_data_losses, 'r');
    legend('training', 'validation');
    ylabel('cost');
    xlabel('iteration number');
    hold off;
  end
  datas2 = {datas.training, datas.validation, datas.test};
  data_names = {'training', 'validation', 'test'};
  for data_i = 1:3,
    data = datas2{data_i};
    data_name = data_names{data_i};
    fprintf('\nThe cost on the %s data is %f\n', data_name, cost(model, data, wd_coefficient));
    if wd_coefficient~=0,
      fprintf('The classification cost (i.e. without weight decay) on the %s data is %f\n', data_name, cost(model, data, 0));
    end
    fprintf('The classification error rate on the %s data is %f\n', data_name, classification_performance(model, data));
  end
end

%% Model
function res = init_model(n_hid)
  n_params = (256+10) * n_hid;
  as_row_vector = cos(0:(n_params-1));
  res = theta_to_model(as_row_vector(:) * 0.1);
end

function res = model_to_theta(model)
  % This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
  input_to_hid_transpose = transpose(model.input_to_hid);
  hid_to_class_transpose = transpose(model.hid_to_class);
  res = [input_to_hid_transpose(:); hid_to_class_transpose(:)];
end

function res = theta_to_model(theta)
  % This function takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
  n_hid = size(theta, 1) / (256+10);
  res.input_to_hid = transpose(reshape(theta(1: 256*n_hid), 256, n_hid));
  res.hid_to_class = reshape(theta(256 * n_hid + 1 : size(theta,1)), n_hid, 10).';
end

function res = cost(model, data, wd_coefficient)
% The function runs the model on the supplied data and returns the cost.
  % model.input_to_hid  Contains the weights from the input units to the hidden units.
  % model.hid_to_class  Contains the weights from the hidden units to the softmax units.
  % data.inputs         Each column corresponds to an input vector. 
  % data.targets        Contains a one-of-N encoding of the target class.
	
  % Forward propagation
  hid_input = model.input_to_hid * data.inputs; 
  hid_output = logistic(hid_input);
  class_input = model.hid_to_class * hid_output;
  log_class_prob = logsoftmax(class_input);
 
  % Compute cost
  classification_loss = -mean(sum(log_class_prob .* data.targets, 1)); % select the right log class probability using that sum; then take the mean over all data cases.
  wd_loss = sum(model_to_theta(model).^2)/2*wd_coefficient; % weight decay cost. very straightforward: E = 1/2 * wd_coeffecient * theta^2
  res = classification_loss + wd_loss;
end

function res = grad(model, data, wd_coefficient)
  % model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
  % model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
  % data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>
  % data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>

  % The returned object is supposed to be exactly like parameter <model>, i.e. it has fields res.input_to_hid and res.hid_to_class. However, the contents of those matrices are gradients (d cost by d model parameter), instead of model parameters.


  % Forward propagation
  hid_input = model.input_to_hid * data.inputs; 
  hid_output = logistic(hid_input);
  class_input = model.hid_to_class * hid_output;
  log_class_prob = logsoftmax(class_input);
  class_prob = exp(log_class_prob); 
  % class_prob is the model output.
  
  %% TODO - Write code here ---------------

    % Right now the function just returns a lot of zeros. Your job is to change that.
    res.input_to_hid = model.input_to_hid * 0;
    res.hid_to_class = model.hid_to_class * 0;
  % ---------------------------------------
end

%% Activation functions
function res = logistic(input)
  res = 1 ./ (1 + exp(-input));
end

function log_class_prob = logsoftmax(class_input)
  % The following lines of code implement the softmax.
  % However, it's written differently from what the lectures say.
  % In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
  % What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable. 
  % "Numerically stable" means that this way, there will never be really big numbers involved.
  % The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Matlab.
  % Matlab isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.
  % This computes log(sum(exp(a), 1)) in a numerically stable way
  
  % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
  maxs_small = max(class_input, [], 1);
  maxs_big = repmat(maxs_small, [size(class_input, 1), 1]);
  class_normalizer = log(sum(exp(class_input - maxs_big), 1)) + maxs_small;
  
  log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
  
end

%% Test and evaluation
function res = classification_performance(model, data)
  % This returns the fraction of data cases that is incorrectly classified by the model.
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  [dump, choices] = max(class_input); % choices is integer: the chosen class, plus 1.
  [dump, targets] = max(data.targets); % targets is integer: the target class, plus 1.
  res = mean(double(choices ~= targets));
end

function test_gradient(model, data, wd_coefficient)
  % Function for doing gradient checking 
  base_theta = model_to_theta(model);
  h = 1e-2;
  correctness_threshold = 1e-5;
  analytic_gradient_struct = grad(model, data, wd_coefficient);
  if size(fieldnames(analytic_gradient_struct), 1) ~= 2,
     error('The object returned by function d_loss_by_d_model should have exactly two field names: .input_to_hid and .hid_to_class');
  end
  if any(size(analytic_gradient_struct.input_to_hid) ~= size(model.input_to_hid)),
     error(sprintf(['The size of .input_to_hid of the return value of d_loss_by_d_model (currently [%d, %d]) should be same as the size of model.input_to_hid (currently [%d, %d])'], size(analytic_gradient_struct.input_to_hid), size(model.input_to_hid)));
  end
  if any(size(analytic_gradient_struct.hid_to_class) ~= size(model.hid_to_class)),
     error(sprintf(['The size of .hid_to_class of the return value of d_loss_by_d_model (currently [%d, %d]) should be same as the size of model.hid_to_class (currently [%d, %d])'], size(analytic_gradient_struct.hid_to_class), size(model.hid_to_class)));
  end
  analytic_gradient = model_to_theta(analytic_gradient_struct);
  if any(isnan(analytic_gradient)) || any(isinf(analytic_gradient)),
     error('Your gradient computation produced a NaN or infinity. That is an error.')
  end
  
  input_to_hid_theta_size = prod(size(model.input_to_hid));
  hid_to_class_theta_size = prod(size(model.hid_to_class));
  big_prime = 1299721; % 1299721 is prime and thus ensures a somewhat random-like selection of indices.
  hid_to_class_indices_to_check = mod(big_prime * (1:20), hid_to_class_theta_size) + 1 + input_to_hid_theta_size;
  input_to_hid_indices_to_check = mod(big_prime * (1:80), input_to_hid_theta_size) + 1;
  indices_to_check = [hid_to_class_indices_to_check, input_to_hid_indices_to_check];
  for i = 1:100,
    test_index = indices_to_check(i);
    analytic_here = analytic_gradient(test_index);
    theta_step = base_theta * 0;
    theta_step(test_index) = h;
    contribution_distances = [-4:-1, 1:4];
    contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280];
    temp = 0;
    for contribution_index = 1:8,
      temp = temp + cost(theta_to_model(base_theta + theta_step * contribution_distances(contribution_index)), data, wd_coefficient) * contribution_weights(contribution_index);
    end
    fd_here = temp / h;
    diff = abs(analytic_here - fd_here);
    % fprintf('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
    if (diff > correctness_threshold) && (diff / (abs(analytic_here) + abs(fd_here)) > correctness_threshold),
      part_names = {'input_to_hid', 'hid_to_class'};
      error(sprintf('Theta element #%d (part of %s), with value %e, has finite difference gradient %e but analytic gradient %e. That looks like an error.\n', test_index, part_names{(i<=20)+1}, base_theta(test_index), fd_here, analytic_here));
    end
    if i==20, fprintf('Gradient test passed for hid_to_class. '); end
    if i==100, fprintf('Gradient test passed for input_to_hid. '); end
  end
  fprintf('Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).\n');
end