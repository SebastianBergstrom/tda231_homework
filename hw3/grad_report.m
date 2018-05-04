%% TODO - Write code here ---------------
    % Right now the function just returns a lot of zeros. Your job is to change that.
    function res = logistic_derivative(input)
        res = exp(-input) ./ (1 + exp(-input)).^2;
    end
    res.input_to_hid = wd_coefficient*model.input_to_hid + ...
                        1/size(data.inputs,2) * model.hid_to_class' * (class_prob-data.targets) ...
                       .* logistic_derivative(model.input_to_hid*data.inputs)  * data.inputs';
    res.hid_to_class = wd_coefficient*model.hid_to_class + ...
                       1/size(data.inputs,2) * (class_prob-data.targets) * logistic(model.input_to_hid * data.inputs)';
  % ---------------------------------------