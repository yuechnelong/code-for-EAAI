function [Mdl2, best_params, Convergence_curve, Loss2] = optimize_fitrTrans_LSTMS1(p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, num_pop, num_iter, method_mti, max_epoch, min_batchsize, lstms_label)
% Optimize Transformer-BiLSTM model parameters
% Input:
% p_train1: Training data features
% train_y_feature_label_norm: Training labels
% p_vaild1: Validation data features
% vaild_y_feature_label_norm: Validation labels
% num_pop: Population size
% num_iter: Number of iterations
% method_mti: Optimization algorithm name
% max_epoch: Maximum training epochs
% min_batchsize: Minimum batch size
% lstms_label: Model type label
% Output:
% Mdl2: Trained model
% best_params: Best hyperparameters [position_encoding_power, attention_heads_power, lstm_neurons_power, dropout_rate, learning_rate]
% Convergence_curve: Convergence curve
% Loss2: Loss value

pop = num_pop;
M = num_iter;
LB = [0, 0, 2, 0.001, 0.001]; % Lower bounds
UB = [6, 6, 8, 0.5, 0.01]; % Upper bounds
% Corresponding to: Position encoding 2^attention, attention heads 2^, LSTM neurons 2^, dropout rate, learning rate

nvars = length(LB);
fit_fitrensemble1 = @fit_fitrTrans_LSTM; % Optimization objective function

% Select optimization algorithm
if strcmp(method_mti, 'GRIME') == 1
    [fMin, Mdl, Convergence_curve, pos] = GRIME_GRIME(pop, M, LB, UB, nvars, fit_fitrensemble1, ...
        p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, ...
        max_epoch, min_batchsize, lstms_label);
end

Mdl2 = Mdl.Mdl;
Loss2 = Mdl.Loss;
best_params = pos; % 返回最佳超参数

% Plot convergence curve
figure
plot(Convergence_curve, '--p', 'LineWidth', 1.2, 'Color', [160, 123, 194] ./ 255)
xticks([1:length(Convergence_curve)])
title('Optimization Process')
xlabel('Iteration Count')
ylabel('Fitness Value')
grid on
set(gca, "FontName", "Times New Roman", "FontSize", 12, "LineWidth", 1.2)
box off

% Display optimization results
if lstms_label == 3
    disp([method_mti, ' Optimized Transformer-BiLSTM: ', ...
        "Position Encoding Vectors:", num2str(2^(round(pos(1)))), ...
        " Attention Heads:", num2str(2^(round(pos(2)))), ...
        " Attention Keys:", num2str(2^(round(pos(2)) + 1)), ...
        " Hidden Layer Neurons (bilstmLayerSizes):", num2str(2^(round(pos(3)))), ...
        ' Dropout Layer Rate: ', num2str((pos(4))), ...
        ' Learning Rate: ', num2str((pos(5)))])
    
    % 创建详细的结构化输出
    hyperparams_detail = struct(...
        'position_encoding_power', pos(1), ...
        'position_encoding_value', 2^(round(pos(1))), ...
        'attention_heads_power', pos(2), ...
        'attention_heads_value', 2^(round(pos(2))), ...
        'attention_keys_value', 2^(round(pos(2)) + 1), ...
        'lstm_neurons_power', pos(3), ...
        'lstm_neurons_value', 2^(round(pos(3))), ...
        'dropout_rate', pos(4), ...
        'learning_rate', pos(5), ...
        'best_fitness', fMin, ...
        'num_population', num_pop, ...
        'num_iterations', num_iter, ...
        'max_epochs', max_epoch, ...
        'batch_size', min_batchsize);
    
    disp('Detailed Hyperparameters:');
    disp(hyperparams_detail);
end
end

%% Transformer-BiLSTM Model Definition and Training Function
function [fitness_value, Mdl1] = fit_fitrTrans_LSTM(pop, p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label)
% Build and train Transformer-BiLSTM model
% Input:
% pop: Parameter vector to optimize
% p_train1: Training data
% train_y_feature_label_norm: Training labels
% p_vaild1: Validation data
% vaild_y_feature_label_norm: Validation labels
% max_epoch: Maximum training epochs
% min_batchsize: Batch size
% lstms_label: Model type
% Output:
% fitness_value: Fitness value
% Mdl1: Model structure

if lstms_label == 3
    % Transformer-BiLSTM model
    layers = [
        sequenceInputLayer(length(p_train1{1, 1}), Name = "input")
        positionEmbeddingLayer(length(p_train1{1, 1}), 2^(round(pop(1))), Name = "pos-emb")
        additionLayer(2, Name = "add")
        selfAttentionLayer(2^(round(pop(2))), 2^(round(pop(2)) + 1))
        dropoutLayer(pop(4)) % Prevent overfitting
        selfAttentionLayer(2^(round(pop(2))), 2^(round(pop(2)) + 1))
        dropoutLayer(pop(4));
        bilstmLayer(2^(round(pop(3))), 'OutputMode', 'sequence') % Bidirectional LSTM layer
        fullyConnectedLayer(length(train_y_feature_label_norm{1, 1}))
        regressionLayer];

    lgraph = layerGraph(layers);
    layers = connectLayers(lgraph, "input", "add/in2");
end

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', max_epoch, ...
    'MiniBatchSize', min_batchsize, ...
    'InitialLearnRate', pop(5), ...
    'ValidationFrequency', 20);

% Train model
[Mdl, Loss] = trainNetwork(p_train1, train_y_feature_label_norm, layers, options);

% Predict on validation set
P_vaild_y_feature_label_norm = predict(Mdl, p_vaild1, 'MiniBatchSize', min_batchsize);

% Calculate fitness value (MAE)
P_vaild_y_feature_label_norm1 = [];
vaild_y_feature_label_norm1 = [];
for i = 1:length(P_vaild_y_feature_label_norm)
    P_vaild_y_feature_label_norm1(i, :) = (P_vaild_y_feature_label_norm{i, 1});
    vaild_y_feature_label_norm1(i, :) = (vaild_y_feature_label_norm{i, 1});
end

fitness_value = sum(sum(abs(P_vaild_y_feature_label_norm1 - vaild_y_feature_label_norm1))) / length(vaild_y_feature_label_norm1);

% Return model
Mdl1.Mdl = Mdl;
Mdl1.Loss = Loss;
end

%% GRIME Algorithm (RIME Algorithm Improved with Mapping and Opposite-Based Learning)
function [Best_rime_rate, Mdl_best, Convergence_curve, Best_rime] = GRIME_GRIME(N, Max_iter, lb, ub, dim, fobj, train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label)
% GRIME optimization algorithm
% Input:
% N: Population size
% Max_iter: Maximum number of iterations
% lb: Lower bounds
% ub: Upper bounds
% dim: Dimension
% fobj: Objective function
% Others: Training and validation data
% Output:
% Best_rime_rate: Best fitness value
% Mdl_best: Best model
% Convergence_curve: Convergence curve
% Best_rime: Best position

% Initialize best position
Best_rime = zeros(1, dim);
Best_rime_rate = inf; % Minimization problem

% Initialize population using chaotic mapping
label = 5; % Use Sine mapping
Rimepop = yinshe(N, dim, label, ub, lb);
Rimepop = min(ub, Rimepop); % Boundary handling
Rimepop = max(lb, Rimepop);

Lb = lb .* ones(1, dim); % Lower bound vector
Ub = ub .* ones(1, dim); % Upper bound vector

it = 1; % Iteration counter
Convergence_curve = zeros(1, Max_iter);
Rime_rates = zeros(1, N); % Fitness values
newRime_rates = zeros(1, N);
W = 5; % Soft rime parameter

% Calculate initial fitness
for i = 1:N
    [Rime_rates(1, i), Mdl_all{1, i}] = fobj(Rimepop(i, :), train_x_feature_label_norm, train_y_feature_label_norm, ...
        vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label);
    
    % Greedy selection
    if Rime_rates(1, i) < Best_rime_rate
        Best_rime_rate = Rime_rates(1, i);
        Best_rime = Rimepop(i, :);
        Mdl_best = Mdl_all{1, i};
    end
end

% Main loop
while it <= Max_iter
    % Calculate rime factor
    RimeFactor = (rand - 0.5) * 2 * cos((pi * it / (Max_iter / 10))) * (1 - round(it * W / Max_iter) / W);
    E = sqrt(it / Max_iter); % Exploration factor
    
    newRimepop = Rimepop; % Record new population
    normalized_rime_rates = normr(Rime_rates); % Normalized fitness
    
    % Soft rime search strategy
    for i = 1:N
        for j = 1:dim
            % Soft rime search
            r1 = rand();
            if r1 < E
                newRimepop(i, j) = Best_rime(1, j) + RimeFactor * ((Ub(j) - Lb(j)) * rand + Lb(j));
            end
            
            % Hard rime puncture mechanism
            r2 = rand();
            if r2 < normalized_rime_rates(i)
                newRimepop(i, j) = Best_rime(1, j);
            end
        end
    end
    
    % Boundary handling
    for i = 1:N
        Flag4ub = newRimepop(i, :) > ub;
        Flag4lb = newRimepop(i, :) < lb;
        newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        
        [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), train_x_feature_label_norm, train_y_feature_label_norm, ...
            vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label);
        
        % Positive greedy selection mechanism
        if newRime_rates(1, i) < Rime_rates(1, i)
            Rime_rates(1, i) = newRime_rates(1, i);
            Rimepop(i, :) = newRimepop(i, :);
            Mdl_all{1, i} = Mdl_new{1, i};
            
            if newRime_rates(1, i) < Best_rime_rate
                Best_rime_rate = Rime_rates(1, i);
                Best_rime = Rimepop(i, :);
                Mdl_best = Mdl_all{1, i};
            end
        end
    end
    
    % Opposite-based learning strategy
    for i = 1:N
        k = (1 + (it / Max_iter)^0.5)^10;
        newRimepop(i, :) = (ub + lb) / 2 + (ub + lb) / (2 * k) - Rimepop(i, :) / k;
        
        Flag4ub = newRimepop(i, :) > ub;
        Flag4lb = newRimepop(i, :) < lb;
        newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        
        [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), train_x_feature_label_norm, train_y_feature_label_norm, ...
            vaild_x_feature_label_norm, vaild_y_feature_label_norm, max_epoch, min_batchsize, lstms_label);
        
        if newRime_rates(1, i) < Rime_rates(1, i)
            Rime_rates(1, i) = newRime_rates(1, i);
            Rimepop(i, :) = newRimepop(i, :);
            Mdl_all{1, i} = Mdl_new{1, i};
            
            if newRime_rates(1, i) < Best_rime_rate
                Best_rime_rate = Rime_rates(1, i);
                Best_rime = Rimepop(i, :);
                Mdl_best = Mdl_all{1, i};
            end
        end
    end
    
    Convergence_curve(it) = Best_rime_rate;
    it = it + 1;
end
end

%% Chaotic Mapping Function
function result = yinshe(N, dim, label, ub, lb)
% Chaotic mapping to generate initial population
% Input:
% N: Population size
% dim: Dimension
% label: Mapping type
% ub: Upper bounds
% lb: Lower bounds
% Output:
% result: Generated population

if label == 1
    % tent mapping
    tent = 1.2; % Tent chaotic coefficient
    Tent = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            if Tent(i, j - 1) < tent
                Tent(i, j) = Tent(i, j - 1) / tent;
            elseif Tent(i, j - 1) >= tent
                Tent(i, j) = (1 - Tent(i, j - 1)) / (1 - tent);
            end
        end
    end
    result = lb + Tent .* (ub - lb);
    
elseif label == 2
    % chebyshev mapping
    chebyshev = 2;
    Chebyshev = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            Chebyshev(i, j) = cos(chebyshev .* acos(Chebyshev(i, j - 1)));
        end
    end
    result = lb + (Chebyshev + 1) / 2 .* (ub - lb);
    
elseif label == 3
    % singer mapping
    u = 1;
    singer = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            singer(i, j) = u * (7.86 * singer(i, j - 1) - 23.31 * singer(i, j - 1).^2 + ...
                28.75 * singer(i, j - 1).^3 - 13.302875 * singer(i, j - 1).^4);
        end
    end
    result = lb + singer .* (ub - lb);
    
elseif label == 4
    % Logistic mapping
    miu = 2; % Chaotic coefficient
    Logistic = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            Logistic(i, j) = miu .* Logistic(i, j - 1) .* (1 - Logistic(i, j - 1));
        end
    end
    result = lb + Logistic .* (ub - lb);
    
elseif label == 5
    % Sine mapping
    sine = 2;
    Sine = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            Sine(i, j) = (4 / sine) * sin(pi * Sine(i, j - 1));
        end
    end
    result = lb + Sine .* (ub - lb);
    
elseif label == 6
    % Circle mapping
    a = 0.5;
    b = 0.6;
    Circle = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            Circle(i, j) = mod(Circle(i, j - 1) + a - b / (2 * pi) * sin(2 * pi * Circle(i, j - 1)), 1);
        end
    end
    result = lb + Circle .* (ub - lb);
else
    % No mapping, random generation
    result = lb + rand(N, dim) .* (ub - lb);
end
end

%% Boundary Handling Function
function s = Bounds(s, Lb, Ub)
% Boundary handling function
% Input:
% s: Vector to be processed
% Lb: Lower bounds
% Ub: Upper bounds
% Output:
% s: Processed vector

% Apply lower bounds
temp = s;
I = temp < Lb;
temp(I) = Lb(I);

% Apply upper bounds
J = temp > Ub;
temp(J) = Ub(J);

% Round first three parameters (position encoding, attention heads, LSTM neurons)
temp(1:3) = round(temp(1:3));
s = temp;
end

%% Initialization Function
function Positions = initialization(SearchAgents_no, dim, ub, lb)
% Initialize population
% Input:
% SearchAgents_no: Number of search agents
% dim: Dimension
% ub: Upper bounds
% lb: Lower bounds
% Output:
% Positions: Initialized positions

Boundary_no = size(ub, 2); % Number of boundaries

% If all variables have the same bounds
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end

% If each variable has different bounds
if Boundary_no > 1
    for i = 1:dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
end
end

%% Helper Function: Matrix Normalization
function Y = normr(X)
% Matrix row normalization
% Input:
% X: Input matrix
% Output:
% Y: Normalized matrix

[m, n] = size(X);
Y = zeros(m, n);
for i = 1:m
    norm_row = norm(X(i, :));
    if norm_row == 0
        Y(i, :) = zeros(1, n);
    else
        Y(i, :) = X(i, :) / norm_row;
    end
end
end

%% Hyperparameter Analysis and Visualization Function
function analyze_hyperparameters(all_hyperparams)
% Analyze and visualize hyperparameter results
% Input:
% all_hyperparams: Cell array or struct containing all hyperparameter results

% Convert to table for easier analysis
if isstruct(all_hyperparams)
    % If it's a struct array
    num_modals = length(all_hyperparams);
    all_data = [];
    
    for i = 1:num_modals
        modal_data = all_hyperparams(i);
        for j = 1:length(modal_data.submodels)
            submodel = modal_data.submodels{j};
            row = [i, j, submodel.position_encoding, submodel.attention_heads, ...
                   submodel.attention_keys, submodel.lstm_neurons, submodel.dropout_rate, ...
                   submodel.learning_rate, submodel.best_fitness, submodel.optimization_time];
            all_data = [all_data; row];
        end
    end
    
    hyperparam_table = array2table(all_data, 'VariableNames', ...
        {'Modal_Index', 'Submodel_Index', 'Position_Encoding', 'Attention_Heads', ...
         'Attention_Keys', 'LSTM_Neurons', 'Dropout_Rate', 'Learning_Rate', ...
         'Best_Fitness', 'Optimization_Time'});
else
    error('Input must be a struct array');
end

% Create visualization figure
figure('Position', [100, 100, 1200, 800]);

% 1. Fitness vs Hyperparameters
subplot(2, 3, 1);
scatter(hyperparam_table.Position_Encoding, hyperparam_table.Best_Fitness, 50, 'filled');
xlabel('Position Encoding Size');
ylabel('Fitness Value');
title('Fitness vs Position Encoding');
grid on;

subplot(2, 3, 2);
scatter(hyperparam_table.LSTM_Neurons, hyperparam_table.Best_Fitness, 50, 'filled');
xlabel('LSTM Neurons');
ylabel('Fitness Value');
title('Fitness vs LSTM Neurons');
grid on;

subplot(2, 3, 3);
scatter(hyperparam_table.Learning_Rate, hyperparam_table.Best_Fitness, 50, 'filled');
xlabel('Learning Rate');
ylabel('Fitness Value');
title('Fitness vs Learning Rate');
grid on;

% 2. Parameter distributions
subplot(2, 3, 4);
histogram(hyperparam_table.Position_Encoding, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('Position Encoding Size');
ylabel('Frequency');
title('Position Encoding Distribution');
grid on;

subplot(2, 3, 5);
histogram(hyperparam_table.LSTM_Neurons, 'FaceColor', [0.8, 0.2, 0.2]);
xlabel('LSTM Neurons');
ylabel('Frequency');
title('LSTM Neurons Distribution');
grid on;

subplot(2, 3, 6);
boxplot(hyperparam_table.Best_Fitness, hyperparam_table.Modal_Index);
xlabel('Modal Index');
ylabel('Fitness Value');
title('Fitness Distribution by Modal');
grid on;

% Display statistics
fprintf('\n=== Hyperparameter Statistics ===\n');
fprintf('Total number of optimizations: %d\n', height(hyperparam_table));
fprintf('Average fitness value: %.6f\n', mean(hyperparam_table.Best_Fitness));
fprintf('Best fitness value: %.6f\n', min(hyperparam_table.Best_Fitness));
fprintf('Worst fitness value: %.6f\n', max(hyperparam_table.Best_Fitness));
fprintf('Average optimization time: %.2f seconds\n', mean(hyperparam_table.Optimization_Time));

% Find best parameters
[best_fitness, best_idx] = min(hyperparam_table.Best_Fitness);
best_row = hyperparam_table(best_idx, :);

fprintf('\n=== Best Hyperparameters ===\n');
fprintf('Modal Index: %d\n', best_row.Modal_Index);
fprintf('Submodel Index: %d\n', best_row.Submodel_Index);
fprintf('Position Encoding: %d\n', best_row.Position_Encoding);
fprintf('Attention Heads: %d\n', best_row.Attention_Heads);
fprintf('Attention Keys: %d\n', best_row.Attention_Keys);
fprintf('LSTM Neurons: %d\n', best_row.LSTM_Neurons);
fprintf('Dropout Rate: %.4f\n', best_row.Dropout_Rate);
fprintf('Learning Rate: %.4f\n', best_row.Learning_Rate);
fprintf('Fitness Value: %.6f\n', best_row.Best_Fitness);
fprintf('Optimization Time: %.2f seconds\n', best_row.Optimization_Time);

% Save analysis results
analysis_results = struct(...
    'hyperparam_table', hyperparam_table, ...
    'statistics', struct(...
        'total_optimizations', height(hyperparam_table), ...
        'average_fitness', mean(hyperparam_table.Best_Fitness), ...
        'best_fitness', min(hyperparam_table.Best_Fitness), ...
        'worst_fitness', max(hyperparam_table.Best_Fitness), ...
        'average_time', mean(hyperparam_table.Optimization_Time) ...
    ), ...
    'best_parameters', table2struct(best_row) ...
);

save('hyperparameter_analysis.mat', 'analysis_results');
fprintf('\nAnalysis results saved to: hyperparameter_analysis.mat\n');
end

%% Example Usage Code
function example_usage()
% Example usage code
% Assume data already exists
% p_train1: Training features
% train_y_feature_label_norm: Training labels
% p_vaild1: Validation features
% vaild_y_feature_label_norm: Validation labels

% Set parameters
num_pop = 30; % Population size
num_iter = 50; % Number of iterations
method_mti = 'GRIME'; % Optimization method
max_epoch = 100; % Maximum training epochs
min_batchsize = 32; % Batch size
lstms_label = 3; % Transformer-BiLSTM model

% Run optimization
[Mdl2, best_params, Convergence_curve, Loss2] = optimize_fitrTrans_LSTMS1(...
    p_train1, train_y_feature_label_norm, p_vaild1, vaild_y_feature_label_norm, ...
    num_pop, num_iter, method_mti, max_epoch, min_batchsize, lstms_label);

% Display results
disp('Optimization completed!');
disp(['Best fitness value: ', num2str(min(Convergence_curve))]);
disp(['Final loss value: ', num2str(Loss2)]);

% Display best hyperparameters
disp('Best Hyperparameters:');
disp(['  Position Encoding Power: ', num2str(best_params(1))]);
disp(['  Attention Heads Power: ', num2str(best_params(2))]);
disp(['  LSTM Neurons Power: ', num2str(best_params(3))]);
disp(['  Dropout Rate: ', num2str(best_params(4))]);
disp(['  Learning Rate: ', num2str(best_params(5))]);

% Calculate actual values
disp('Actual Parameter Values:');
disp(['  Position Encoding: ', num2str(2^(round(best_params(1))))]);
disp(['  Attention Heads: ', num2str(2^(round(best_params(2))))]);
disp(['  Attention Keys: ', num2str(2^(round(best_params(2)) + 1))]);
disp(['  LSTM Neurons: ', num2str(2^(round(best_params(3))))]);

% Save results to file
results = struct(...
    'model', Mdl2, ...
    'best_hyperparameters', best_params, ...
    'convergence_curve', Convergence_curve, ...
    'loss', Loss2, ...
    'actual_parameters', struct(...
        'position_encoding', 2^(round(best_params(1))), ...
        'attention_heads', 2^(round(best_params(2))), ...
        'attention_keys', 2^(round(best_params(2)) + 1), ...
        'lstm_neurons', 2^(round(best_params(3))), ...
        'dropout_rate', best_params(4), ...
        'learning_rate', best_params(5) ...
    ) ...
);

save('optimization_results.mat', 'results');
disp('Results saved to: optimization_results.mat');
end