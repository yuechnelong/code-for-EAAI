clc;clear;close all;	
load('data.mat')	
random_seed=1;	
rng(random_seed)		
data_str='data.xlsx'
dataO1=readtable(data_str,'VariableNamingRule','preserve');  	
data1=dataO1(:,2:end);test_data=table2cell(dataO1(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;    	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;    	
      else	
        index_la(i)=0;     	
    end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
    %% 
if length(index_double)>=1	
     data_numshuju=table2array(data1(:,index_double));	
     data_numshuju2=data_numshuju;	
     index_need_last=index_double;	
else	
     index_need_last=index_double;	
    data_numshuju2=[];	
end		
data_shuju=[];	
if length(index_char)>=1	
   for j=1:length(index_char)	
     data_get=table2array(data1(:,index_char(j)));	
	
     data_label=unique(data_get);	
     for NN=1:length(data_label)	
         idx = find(ismember(data_get,data_label{NN,1}));	
         data_shuju(idx,j)=NN;	
     end	
   end	
end	
data_all_last=[data_shuju,data_numshuju2];	
label_all_last=[index_char,index_need_last];	
data=data_all_last;	
     data_biao_all=data1.Properties.VariableNames;	
for j=1:length(label_all_last)	
     data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
end	
    %% 
	
[data, index_ab_unique] = abnorm_detect(data);
	
    %% 
data_numshuju=data;	
for NN=1:size(data_numshuju,2)	
      data_test=data_numshuju(:,NN);	
      index=isnan(data_test);	
      data_test1=data_test;	
      data_test1(index)=[];	
      index_label=1:length(data_test);	
      index_label1=index_label;	
      index_label1(index)=[];	
     data_all=interp1(index_label1,data_test1,index_label,'spline');	
     dataO(:,NN)=data_all;	
end	
%% 
A_data1 = dataO;
data_biao1 = data_biao;
select_feature_num = 10;   
predict_num = 1;           
index_name = data_biao1;
print_index_name = [];
num_pop = 20;              
num_iter = 40;             
method_mti = 'GRIME';

fprintf('Initiating stability feature selection analysis...\n');

n_runs = 10;               
stability_threshold = 0.6; 

all_selected_features = cell(n_runs, 1);
all_convergence_curves = zeros(num_iter, n_runs);
all_fitness_values = zeros(n_runs, 1);
run_times = zeros(n_runs, 1);

% Serial execution
fprintf('Performing %d GRIME runs...\n', n_runs);
for run_idx = 1:n_runs
    t_start = tic;
    
    % Set random seed for reproducibility
    rng(run_idx * 1000 + 12345);
    
    fprintf('Performing GRIME run %d/%d...\n', run_idx, n_runs);
    
    % Execute GRIME optimization
    [pos_get, ~, fMin1, Convergence_curve1] = ...
        optimize_fitrfeature_selected1(A_data1, predict_num, ...
                                      num_pop, num_iter, method_mti);
    
    % Store results
    all_selected_features{run_idx} = pos_get;
    all_fitness_values(run_idx) = fMin1;
    if length(Convergence_curve1) >= num_iter
        all_convergence_curves(:, run_idx) = Convergence_curve1(1:num_iter)';
    else
        all_convergence_curves(1:length(Convergence_curve1), run_idx) = Convergence_curve1(:);
    end
    
    % Record runtime
    run_times(run_idx) = toc(t_start);
    fprintf('GRIME run %d/%d completed, time: %.2f seconds\n', run_idx, n_runs, run_times(run_idx));
end
fprintf('\nCalculating stability metrics...\n');


n_total_features = size(A_data1, 2) - predict_num;
feature_counts = zeros(1, n_total_features);

for run_idx = 1:n_runs
    features = all_selected_features{run_idx};
    if ~isempty(features)
        feature_counts(features) = feature_counts(features) + 1;
    end
end

feature_frequency = feature_counts / n_runs;


jaccard_similarities = zeros(n_runs, n_runs);
for i = 1:n_runs-1
    for j = i+1:n_runs
        set1 = all_selected_features{i};
        set2 = all_selected_features{j};
        
        if ~isempty(set1) && ~isempty(set2)
            intersection = length(intersect(set1, set2));
            union_size = length(union(set1, set2));
            
            if union_size > 0
                jaccard_similarities(i, j) = intersection / union_size;
                jaccard_similarities(j, i) = jaccard_similarities(i, j);
            end
        end
    end
end


pairwise_jaccard = [];
for i = 1:n_runs-1
    for j = i+1:n_runs
        if jaccard_similarities(i, j) > 0
            pairwise_jaccard = [pairwise_jaccard, jaccard_similarities(i, j)];
        end
    end
end

if ~isempty(pairwise_jaccard)
    avg_jaccard = mean(pairwise_jaccard);
    std_jaccard = std(pairwise_jaccard);
else
    avg_jaccard = 0;
    std_jaccard = 0;
end


selection_sizes = zeros(1, n_runs);
for i = 1:n_runs
    selection_sizes(i) = length(all_selected_features{i});
end
size_mean = mean(selection_sizes);
size_std = std(selection_sizes);
size_cv = size_std / (size_mean + eps);


selected_features = find(feature_frequency >= stability_threshold);


if isempty(selected_features)
    k = min(select_feature_num, floor(n_total_features * 0.3));
    [~, sorted_idx] = sort(feature_frequency, 'descend');
    selected_features = sorted_idx(1:k);
    fprintf('No features were selected using frequency threshold %.2f, switching to Top-%d features.\n', stability_threshold, k);
end

pos_get = selected_features;
feature_need_last = pos_get;


fprintf('\n========== Stability Analysis Results ==========\n');
fprintf('Total number of runs: %d\n', n_runs);
fprintf('Average running time: %.2f ± %.2f seconds\n', mean(run_times), std(run_times));
fprintf('Average fitness value: %.4f ± %.4f\n', mean(all_fitness_values), std(all_fitness_values));
fprintf('\nStability Metrics:\n');
fprintf('Average Jaccard similarity: %.4f ± %.4f\n', avg_jaccard, std_jaccard);
fprintf('Feature selection size: %.1f ± %.1f (Coefficient of variation: %.3f)\n', size_mean, size_std, size_cv);
fprintf('Final number of selected features: %d\n', length(selected_features));
fprintf('Feature selection threshold: %.2f\n', stability_threshold);

for NN = 1:length(pos_get)
    print_index_name{1, NN} = index_name{1, pos_get(NN)};
end

fprintf('\nFinal selected features:\n');
disp(print_index_name);

fprintf('\nFeature selection frequencies:\n');
for NN = 1:length(print_index_name)
    fprintf('%s: %.1f%%\n', print_index_name{1, NN}, feature_frequency(pos_get(NN))*100);
end

data_select = [A_data1(:, pos_get), A_data1(:, end-predict_num+1:end)];
feature_need_last = pos_get;

figure('Position', [100, 100, 1200, 600], 'Name', 'GRIME Feature Selection Stability Analysis');


subplot(1, 3, 1);
hold on;

for run_idx = 1:min(15, n_runs)
    plot(all_convergence_curves(:, run_idx), 'Color', [0.7, 0.7, 0.7, 0.3], 'LineWidth', 0.5);
end

mean_curve = mean(all_convergence_curves, 2);
std_curve = std(all_convergence_curves, 0, 2);
plot(mean_curve, 'b-', 'LineWidth', 2.5);
plot(mean_curve + std_curve, 'r--', 'LineWidth', 1.5);
plot(mean_curve - std_curve, 'r--', 'LineWidth', 1.5);
xlabel('Iteration Count');
ylabel('Fitness Value');
title('GRIME Convergence Curve (Multiple Runs)');
legend('Single Run', 'Average Curve', '±Standard Deviation', 'Location', 'best');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1.2);
box off;

subplot(1, 3, 2);
bar(feature_frequency, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
plot([0, n_total_features+1], [stability_threshold, stability_threshold], ...
     'r--', 'LineWidth', 2);
xlabel('Feature Index');
ylabel('Selection Frequency');
title(sprintf('Feature Selection Frequency Distribution\n(Threshold = %.2f)', stability_threshold));
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1.2);
box off;

subplot(1, 3, 3);
histogram(selection_sizes, 10, 'FaceColor', [0.4, 0.8, 0.4], 'EdgeColor', 'k');
hold on;
plot([size_mean, size_mean], [0, max(histcounts(selection_sizes, 10))], ...
     'r-', 'LineWidth', 2);
xlabel('Number of Selected Features');
ylabel('Frequency');
title(sprintf('Feature Selection Size Distribution\nMean = %.1f, Std = %.1f', size_mean, size_std));
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1.2);
box off;

figure;
plot(mean_curve, '--*', 'LineWidth', 1.2, 'Color', [0.4784, 0.7412, 0.3373]);
xticks(1:5:length(mean_curve));
title('GRIME Feature Selection Optimization Process (Average Convergence Curve)');
xlabel('Iteration Count');
ylabel('Fitness Value');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
box off;

data_select1 = data_select;

% ==================== Save Analysis Results ====================
stability_results = struct();
stability_results.n_runs = n_runs;
stability_results.selected_features = selected_features;
stability_results.feature_names = print_index_name;
stability_results.feature_frequency = feature_frequency;
stability_results.avg_jaccard = avg_jaccard;
stability_results.std_jaccard = std_jaccard;
stability_results.size_mean = size_mean;
stability_results.size_std = size_std;
stability_results.size_cv = size_cv;
stability_results.all_fitness_values = all_fitness_values;
stability_results.convergence_curves = all_convergence_curves;
stability_results.run_times = run_times;
stability_results.data_select = data_select;

save('stability_analysis_results.mat', 'stability_results');
fprintf('\nStability analysis results have been saved to stability_analysis_results.mat\n');

disp('Selected Features:');
disp(print_index_name);

%% 
figure
t = 1:length(data_select1(:, end));
num_pop = 20;
num_iter = 40;
method_mti = 'GRIME';


[IMF, Convergence_curve, fMin, pos] = optimize_VMD(data_select1(:, end), num_pop, num_iter, method_mti);
IMF = IMF';

imf = IMF(:, 1:end-1);
res = IMF(:, end);
imf_L = [data_select1(:, end), IMF];

[p, q] = ndgrid(t, 1:size(imf, 2));

P = [p, p(:, 1:2)];
Q = [q(:, 1), q+1, q(:, end)+2];
plot3(P, Q, imf_L)
grid on
xlabel('Time Values')
ylabel('Mode Number')
zlabel('Mode Amplitude')

decom_str{1, 1} = 'origin data';
for i = 1:size(imf, 2)
    decom_str{1, i+1} = ['imf', num2str(i)];
end
decom_str{1, 2+size(imf, 2)} = 'res';
yticks(1:length(decom_str))
yticklabels(decom_str)

for NN1 = 1:size(IMF, 2)
    data_select1_cell{1, NN1} = [data_select1(:, 1:end-1), IMF(:, NN1)];
end

plotpl(data_select1(:, end), [imf, res]')


disp('--- Debug Information ---');
disp(['Convergence curve data length: ', num2str(length(Convergence_curve))]);
disp(['Minimum fMin: ', num2str(fMin)]);
figure;
plot(Convergence_curve, '-o', 'LineWidth', 1.5);  % Use simple line style for testing
title('Optimization Convergence Curve');
xlabel('Iteration');
ylabel('Fitness');
grid on;
drawnow;
	
  
	
 %% 
%% 
select_predict_num=6;  
num_feature=3;   
num_series=3;     
num_input_serise=10;     
min_batchsize=432; 
num_pop=20;
num_iter=40;      
max_epoch_LC=70; 
method_mti='GRIME'; 
list_cell= G_out_data.list_cell;

attention_label=0;
attention_head=G_out_data.attention_head;

%% Start timing - Total runtime
fprintf('Starting model training...\n');
totalStartTime = tic;

%% Model training
hyperparameter_results = struct();

x_mu_all = []; x_sig_all = []; y_mu_all = []; y_sig_all = [];
for NUM_all = 1:length(data_select1_cell)
    fprintf('Processing data component %d/%d...\n', NUM_all, length(data_select1_cell));
    data_process = data_select1_cell{1, NUM_all};
    
    [x_feature_label, y_feature_label] = timeseries_process1(data_process, select_predict_num, num_feature, num_series);
    [~, y_feature_label1] = timeseries_process1(data_select1, select_predict_num, num_feature, num_series); % Before decomposition

    index_label1 = 1:(size(x_feature_label, 1)); 
    index_label = index_label1;
    
    spilt_ri = G_out_data.spilt_ri;
    train_num = round(spilt_ri(1) / (sum(spilt_ri)) * size(x_feature_label, 1)); % Number of training samples
    vaild_num = round((spilt_ri(1) + spilt_ri(2)) / (sum(spilt_ri)) * size(x_feature_label, 1)); % Number of validation samples
    
    % Training set, validation set, test set
    train_x_feature_label = x_feature_label(index_label(1:train_num), :);
    train_y_feature_label = y_feature_label(index_label(1:train_num), :);
    vaild_x_feature_label = x_feature_label(index_label(train_num+1:vaild_num), :);
    vaild_y_feature_label = y_feature_label(index_label(train_num+1:vaild_num), :);
    test_x_feature_label = x_feature_label(index_label(vaild_num+1:end), :);
    test_y_feature_label = y_feature_label(index_label(vaild_num+1:end), :);
    
    % Z-score normalization
    
    % Training set
    x_mu = mean(train_x_feature_label);
    x_sig = std(train_x_feature_label);
    train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig; % Normalize training data
    y_mu = mean(train_y_feature_label);
    y_sig = std(train_y_feature_label);
    train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig; % Normalize training data
    
    x_mu_all(NUM_all, :) = x_mu;
    x_sig_all(NUM_all, :) = x_sig;
    y_mu_all(NUM_all, :) = y_mu;
    y_sig_all(NUM_all, :) = y_sig;
    
    % Validation set
    vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig; % Normalize validation data
    vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig; % Normalize validation data
    
    % Test set
    test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig; % Normalize test data
    test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig; % Normalize test data

    y_train_predict_norm = zeros(size(train_y_feature_label, 1), size(train_y_feature_label, 2));
    y_vaild_predict_norm = zeros(size(vaild_y_feature_label, 1), size(vaild_y_feature_label, 2));
    y_test_predict_norm = zeros(size(test_y_feature_label, 1), size(test_y_feature_label, 2));
    
    
    current_modal_hyperparams = cell(1, length(list_cell));

    for N1 = 1:length(list_cell)
        fprintf('  Training sub-model %d/%d...\n', N1, length(list_cell));
        submodelStartTime = tic;
        
        hidden_size = G_out_data.hidden_size;
        p_train1 = cell(size(train_x_feature_label, 1), 1);
        p_test1 = cell(size(test_x_feature_label, 1), 1);
        p_vaild1 = cell(size(vaild_x_feature_label, 1), 1);
        O_train1 = cell(size(train_x_feature_label, 1), 1);
        O_test1 = cell(size(test_x_feature_label, 1), 1);
        O_vaild1 = cell(size(vaild_x_feature_label, 1), 1);
        
        for i = 1:size(train_x_feature_label, 1) % Convert input to cell format
            p_train1{i, 1} = (train_x_feature_label_norm(i, :))';
        end
        for i = 1:size(test_x_feature_label, 1)
            p_test1{i, 1} = (test_x_feature_label_norm(i, :))';
        end
        for i = 1:size(vaild_x_feature_label, 1)
            p_vaild1{i, 1} = (vaild_x_feature_label_norm(i, :))';
        end

        for i = 1:size(train_x_feature_label, 1) % Convert input to cell format
            O_train1{i, 1} = (train_y_feature_label_norm(i, list_cell{1, N1}))';
        end
        for i = 1:size(test_x_feature_label, 1)
            O_test1{i, 1} = (test_y_feature_label_norm(i, list_cell{1, N1}))';
        end
        for i = 1:size(vaild_x_feature_label, 1)
            O_vaild1{i, 1} = (vaild_y_feature_label_norm(i, list_cell{1, N1}))';
        end

        method_mti = 'GRIME';

        % Optimize and train model
        fprintf('    Starting model optimization...\n');
        optimizeStartTime = tic;
        [Mdl, best_params, fitness, Loss] = optimize_fitrTrans_LSTMS1(p_train1, O_train1, p_vaild1, O_vaild1, num_pop, num_iter, method_mti, max_epoch_LC, min_batchsize, 3);
        optimizeTime = toc(optimizeStartTime);
        fprintf('    Model optimization completed, time: %.2f seconds\n', optimizeTime);
        
        
        current_modal_hyperparams{N1} = struct(...
            'submodel_index', N1, ...
            'best_parameters', best_params, ...
            'best_fitness', fitness, ...
            'position_encoding', 2^(round(best_params(1))), ...
            'attention_heads', 2^(round(best_params(2))), ...
            'attention_keys', 2^(round(best_params(2)) + 1), ...
            'lstm_neurons', 2^(round(best_params(3))), ...
            'dropout_rate', best_params(4), ...
            'learning_rate', best_params(5), ...
            'optimization_time', optimizeTime);

        % Prediction
        fprintf('    Starting prediction...\n');
        predictStartTime = tic;
        y_train_predict_norm1 = predict(Mdl, p_train1, 'MiniBatchSize', min_batchsize);
        y_vaild_predict_norm1 = predict(Mdl, p_vaild1, 'MiniBatchSize', min_batchsize);
        y_test_predict_norm1 = predict(Mdl, p_test1, 'MiniBatchSize', min_batchsize);
        predictTime = toc(predictStartTime);
        fprintf('    Prediction completed, time: %.2f seconds\n', predictTime);

        y_train_predict_norm_roll = [];
        y_vaild_predict_norm_roll = [];
        y_test_predict_norm_roll = [];

        for i = 1:length(y_train_predict_norm1)
            y_train_predict_norm_roll(i, :) = (y_train_predict_norm1{i, 1});
        end
        for i = 1:length(y_vaild_predict_norm1)
            y_vaild_predict_norm_roll(i, :) = (y_vaild_predict_norm1{i, 1});
        end
        for i = 1:length(y_test_predict_norm1)
            y_test_predict_norm_roll(i, :) = (y_test_predict_norm1{i, 1});
        end
        
        y_train_predict_norm(:, list_cell{1, N1}) = y_train_predict_norm_roll;
        y_vaild_predict_norm(:, list_cell{1, N1}) = y_vaild_predict_norm_roll;
        y_test_predict_norm(:, list_cell{1, N1}) = y_test_predict_norm_roll;
        Model{1, N1} = Mdl;
        model_all{NUM_all, N1} = Mdl;
        graph = layerGraph(Mdl.Layers);
        figure;
        plot(graph);

        % Record sub-model training time
        submodelTime = toc(submodelStartTime);
        fprintf('  Sub-model %d training completed, total time: %.2f seconds\n\n', N1, submodelTime);

        figure
        subplot(2, 1, 1)
        plot(1:length(Loss.TrainingRMSE), Loss.TrainingRMSE, '-', 'LineWidth', 1)
        xlabel('Iteration Count'); ylabel('Root Mean Square Error'); legend('Training Set RMSE'); title('Training Set RMSE Curve'); grid; set(gcf, 'color', 'w')

        subplot(2, 1, 2)
        plot(1:length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1)
        xlabel('Iteration Count'); ylabel('Loss Function'); legend('Training Set Loss'); title('Training Set Loss Function Curve'); grid; set(gcf, 'color', 'w')
    end
    
 
    hyperparameter_results(NUM_all).modal_index = NUM_all;
    hyperparameter_results(NUM_all).submodels = current_modal_hyperparams;
    hyperparameter_results(NUM_all).normalization_params = struct(...
        'x_mu', x_mu, ...
        'x_sig', x_sig, ...
        'y_mu', y_mu, ...
        'y_sig', y_sig);
    
    fprintf('Data component %d processing completed\n\n', NUM_all);
    
    y_train_predict_cell{1, NUM_all} = y_train_predict_norm .* y_sig + y_mu; % Inverse normalization
    y_vaild_predict_cell{1, NUM_all} = y_vaild_predict_norm .* y_sig + y_mu;
    y_test_predict_cell{1, NUM_all} = y_test_predict_norm .* y_sig + y_mu;
end


save('hyperparameter_results.mat', 'hyperparameter_results');


disp('=== Hyperparameter Optimization Results Summary ===');
for i = 1:length(hyperparameter_results)
    fprintf('\nModal %d:\n', i);
    for j = 1:length(hyperparameter_results(i).submodels)
        params = hyperparameter_results(i).submodels{j};
        fprintf('  Sub-model %d:\n', j);
        fprintf('    Position Encoding: %d\n', params.position_encoding);
        fprintf('    Attention Heads: %d\n', params.attention_heads);
        fprintf('    Attention Keys: %d\n', params.attention_keys);
        fprintf('    LSTM Neurons: %d\n', params.lstm_neurons);
        fprintf('    Dropout Rate: %.4f\n', params.dropout_rate);
        fprintf('    Learning Rate: %.4f\n', params.learning_rate);
        fprintf('    Best Fitness: %.6f\n', params.best_fitness);
        fprintf('    Optimization Time: %.2f seconds\n', params.optimization_time);
    end
end


hyperparam_table = [];
for i = 1:length(hyperparameter_results)
    for j = 1:length(hyperparameter_results(i).submodels)
        params = hyperparameter_results(i).submodels{j};
        row = [i, j, params.position_encoding, params.attention_heads, ...
               params.attention_keys, params.lstm_neurons, params.dropout_rate, ...
               params.learning_rate, params.best_fitness, params.optimization_time];
        hyperparam_table = [hyperparam_table; row];
    end
end


csv_filename = 'hyperparameter_results_summary.csv';
csv_header = {'Modal_Index', 'Submodel_Index', 'Position_Encoding', 'Attention_Heads', ...
              'Attention_Keys', 'LSTM_Neurons', 'Dropout_Rate', 'Learning_Rate', ...
              'Best_Fitness', 'Optimization_Time_Seconds'};
fid = fopen(csv_filename, 'w');
fprintf(fid, '%s\n', strjoin(csv_header, ','));
fclose(fid);
dlmwrite(csv_filename, hyperparam_table, '-append', 'delimiter', ',', 'precision', '%.6f');
fprintf('\nHyperparameter results saved to: %s\n', csv_filename);

%% Model Ensemble
fprintf('Starting model ensemble...\n');
ensembleStartTime = tic;

y_train_predict = 0;
y_vaild_predict = 0;
y_test_predict = 0;
for i = 1:length(data_select1_cell)
    y_train_predict = y_train_predict + y_train_predict_cell{1, i};
    y_vaild_predict = y_vaild_predict + y_vaild_predict_cell{1, i};
    y_test_predict = y_test_predict + y_test_predict_cell{1, i};
end

ensembleTime = toc(ensembleStartTime);
fprintf('Model ensemble completed, time: %.2f seconds\n\n', ensembleTime);
%% Performance evaluation
fprintf('Starting performance evaluation...\n');
evalStartTime = tic;

train_y_feature_label = y_feature_label1(index_label(1:train_num), :); 
vaild_y_feature_label = y_feature_label1(index_label(train_num+1:vaild_num), :);
test_y_feature_label = y_feature_label1(index_label(vaild_num+1:end), :);
Tvalue = G_out_data.Tvalue;  % Method used
train_y = train_y_feature_label; 
train_MAE = sum(sum(abs(y_train_predict - train_y))) / size(train_y,1) / size(train_y,2); 
disp([Tvalue, 'Training set mean absolute error MAE: ', num2str(train_MAE)])
train_MAPE = sum(sum(abs((y_train_predict - train_y) ./ train_y))) / size(train_y,1) / size(train_y,2); 
disp([Tvalue, 'Training set mean absolute percentage error MAPE: ', num2str(train_MAPE)])
train_MSE = (sum(sum(((y_train_predict - train_y)).^2)) / size(train_y,1) / size(train_y,2)); 
disp([Tvalue, 'Training set mean squared error MSE: ', num2str(train_MSE)])    
train_RMSE = sqrt(sum(sum(((y_train_predict - train_y)).^2)) / size(train_y,1) / size(train_y,2)); 
disp([Tvalue, 'Training set root mean squared error RMSE: ', num2str(train_RMSE)]) 
train_R2 = 1 - mean(norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   
disp([Tvalue, 'Training set R-squared coefficient R2: ', num2str(train_R2)]) 
disp('************************************************************************************')
vaild_y = vaild_y_feature_label;
vaild_MAE = sum(sum(abs(y_vaild_predict - vaild_y))) / size(vaild_y,1) / size(vaild_y,2); 
disp([Tvalue, 'Validation set mean absolute error MAE: ', num2str(vaild_MAE)])
vaild_MAPE = sum(sum(abs((y_vaild_predict - vaild_y) ./ vaild_y))) / size(vaild_y,1) / size(vaild_y,2); 
disp([Tvalue, 'Validation set mean absolute percentage error MAPE: ', num2str(vaild_MAPE)])
vaild_MSE = (sum(sum(((y_vaild_predict - vaild_y)).^2)) / size(vaild_y,1) / size(vaild_y,2)); 
disp([Tvalue, 'Validation set mean squared error MSE: ', num2str(vaild_MSE)])     
vaild_RMSE = sqrt(sum(sum(((y_vaild_predict - vaild_y)).^2)) / size(vaild_y,1) / size(vaild_y,2)); 
disp([Tvalue, 'Validation set root mean squared error RMSE: ', num2str(vaild_RMSE)]) 
vaild_R2 = 1 - mean(norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);   
disp([Tvalue, 'Validation set R-squared coefficient R2: ', num2str(vaild_R2)]) 
disp('************************************************************************************')
test_y = test_y_feature_label;
test_MAE = sum(sum(abs(y_test_predict - test_y))) / size(test_y,1) / size(test_y,2); 
disp([Tvalue, 'Test set mean absolute error MAE: ', num2str(test_MAE)])
test_MAPE = sum(sum(abs((y_test_predict - test_y) ./ test_y))) / size(test_y,1) / size(test_y,2); 
disp([Tvalue, 'Test set mean absolute percentage error MAPE: ', num2str(test_MAPE)])
test_MSE = (sum(sum(((y_test_predict - test_y)).^2)) / size(test_y,1) / size(test_y,2)); 
disp([Tvalue, 'Test set mean squared error MSE: ', num2str(test_MSE)]) 
test_RMSE = sqrt(sum(sum(((y_test_predict - test_y)).^2)) / size(test_y,1) / size(test_y,2)); 
disp([Tvalue, 'Test set root mean squared error RMSE: ', num2str(test_RMSE)]) 
test_R2 = 1 - mean(norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   
disp([Tvalue, 'Test set R-squared coefficient R2: ', num2str(test_R2)]) 

evalTime = toc(evalStartTime);
fprintf('Performance evaluation completed, time: %.2f seconds\n\n', evalTime);%% Main function: Optimize Transformer-BiLSTM model




%% Final time statistics


totalElapsedTime = toc(totalStartTime);

fprintf('\n========== Time Statistics ==========\n');
fprintf('Total runtime: %.2f seconds\n', totalElapsedTime);
fprintf('Approximately: %.2f minutes\n', totalElapsedTime/60);
fprintf('Approximately: %.2f hours\n\n', totalElapsedTime/3600);

fprintf('Detailed time breakdown:\n');
fprintf('1. Data processing and preprocessing\n');
fprintf('2. Model training and optimization\n');
fprintf('3. Model ensemble: %.2f seconds\n', ensembleTime);
fprintf('4. Performance evaluation: %.2f seconds\n', evalTime);
fprintf('================================\n');

%% Optional: Save time information to workspace
timeInfo = struct();
timeInfo.totalSeconds = totalElapsedTime;
timeInfo.totalMinutes = totalElapsedTime/60;
timeInfo.totalHours = totalElapsedTime/3600;
timeInfo.ensembleTime = ensembleTime;
timeInfo.evaluationTime = evalTime;
timeInfo.endTime = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
assignin('base', 'timeInfo', timeInfo);

fprintf('Training completed at: %s\n', timeInfo.endTime);
	
%% Call model to get prediction results

y_last_predict_cell = cell(1, length(data_select1_cell));
for NUM_all = 1:length(data_select1_cell)
    data_process = data_select1_cell{1,NUM_all};
    data_process = data_process(vaild_num+1:end, :);
    
    [x_feature_label] = timeseries_process1_Pre(data_process, select_predict_num, num_feature, num_series);
    
    x_mu = x_mu_all(NUM_all, :);
    x_sig = x_sig_all(NUM_all, :);
    pre_x_feature_label_norm = (x_feature_label - x_mu) ./ x_sig;    % Training data normalization
    
    for i = 1:size(pre_x_feature_label_norm,1)
        p_pre1{i, 1} = (pre_x_feature_label_norm(i, :))';
    end
    
    for N1 = 1:length(list_cell)
        Mdl = model_all{NUM_all, N1};
        y_pre_predict_norm1 = predict(Mdl, p_pre1, 'MiniBatchSize', min_batchsize);
        y_pre_predict_norm1_roll = [];
        for i = 1:length(y_pre_predict_norm1)
            y_pre_predict_norm1_roll(i, :) = (y_pre_predict_norm1{i, 1});
        end
        y_pre_predict_norm(:, list_cell{1,N1}) = y_pre_predict_norm1_roll;
    end
    
    y_last_predict_cell{1,NUM_all} = y_pre_predict_norm .* y_sig_all(NUM_all,:) + y_mu_all(NUM_all,:);  % Denormalization
end

y_last_predict = 0;   y_before = 0;
for i = 1:length(data_select1_cell)
    y_last_predict = y_last_predict + y_last_predict_cell{1,i};
    y_before = y_before + data_select1_cell{1,i}(vaild_num+1:end, end);
end

y_last_predict1 = y_last_predict(end, 1:end);
disp('Predicted future time point data:')
disp(y_last_predict1)
figure;
plot(1:length(y_before), y_before, '-o', 'LineWidth', 1);
hold on
plot(length(y_before)+1:length(y_before)+length(y_last_predict1), y_last_predict1, '-p', 'LineWidth', 1.2);
hold on
legend('True', 'Predict')
set(gca, 'LineWidth', 1.2)

show_wei_num = 6; % Number of output dimensions to display
train_y1 = train_y(:, show_wei_num);
vaild_y1 = vaild_y(:, show_wei_num);
test_y1 = test_y(:, show_wei_num);

y_train_predict1 = y_train_predict(:, show_wei_num);
y_vaild_predict1 = y_vaild_predict(:, show_wei_num);
y_test_predict1 = y_test_predict(:, show_wei_num);

color_list = G_out_data.color_list;
Line_Width = G_out_data.Line_Width;
rand_list1 = G_out_data.rand_list1;
makesize = G_out_data.makesize;
color_index = G_out_data.color_index;
show_num1 = G_out_data.show_num1;
show_num2 = G_out_data.show_num2;
show_num3 = G_out_data.show_num3;
yang_str2 = G_out_data.yang_str2;
yang_str3 = G_out_data.yang_str3;
yangsi_idnex = G_out_data.yangsi_idnex;
yang_fu3_ku = G_out_data.yang_fu3_ku;
if show_num1 > length(train_y1)
    show_num1 = length(train_y1);
end
index_show = 1:show_num1;
figure_density(yangsi_idnex(5), y_train_predict1(index_show), train_y1(index_show), 'Training Set')
figure('Position', [200,200,600,350]);
XX = 1:length(train_y1);
plot(gca, XX(index_show), train_y1(index_show), yang_fu3_ku{1,yangsi_idnex(1)}, 'Color', color_list(yangsi_idnex(3),:), 'LineWidth', Line_Width(1))
hold(gca, 'on')
plot(gca, XX(index_show), y_train_predict1(index_show), yang_fu3_ku{1,yangsi_idnex(2)}, 'Color', color_list(yangsi_idnex(4),:), 'LineWidth', Line_Width(1), 'MarkerSize', makesize)
hold(gca, 'on')
title('Training Set Testing Results')

FontName = G_out_data.FontName;
FontSize = G_out_data.FontSize;
kuang_width = G_out_data.kuang_width;
set(gca, 'FontName', FontName, 'FontSize', FontSize, 'LineWidth', kuang_width)

xlabel1 = G_out_data.xlabel1;
ylabel1 = G_out_data.ylabel1;
legend1 = G_out_data.legend1;
xlabel(xlabel1)
ylabel(ylabel1)
legend(legend1)

box1 = G_out_data.box1;
box(gca, box1)
le_kuang = G_out_data.le_kuang;
legend(gca, le_kuang) % Remove legend box
grid1 = G_out_data.grid1;
grid(gca, grid1)

if show_num2 > length(test_y1)
    show_num2 = length(test_y1);
end

index_show = 1:show_num2;
figure_density(yangsi_idnex(5), y_test_predict1(index_show), test_y1(index_show), 'Test Set')
figure('Position', [200,200,600,350]);
XX = 1:length(test_y1);
plot(gca, XX(index_show), test_y1(index_show), yang_fu3_ku{1,yangsi_idnex(1)}, 'Color', color_list(yangsi_idnex(3),:), 'LineWidth', Line_Width(1))
hold(gca, 'on')
plot(gca, XX(index_show), y_test_predict1(index_show), yang_fu3_ku{1,yangsi_idnex(2)}, 'Color', color_list(yangsi_idnex(4),:), 'LineWidth', Line_Width(1), 'MarkerSize', makesize)
hold(gca, 'on')
title('Test Set Testing Results')

set(gca, 'FontName', FontName, 'FontSize', FontSize, 'LineWidth', kuang_width)

xlabel(xlabel1)
ylabel(ylabel1)
legend(legend1)

box(gca, box1)
legend(gca, le_kuang) % Remove legend box
grid(gca, grid1)

if show_num3 > length(vaild_y1)
    show_num3 = length(vaild_y1);
end
index_show = 1:show_num3;
figure_density(yangsi_idnex(5), y_vaild_predict1(index_show), vaild_y1(index_show), 'Validation Set')
figure('Position', [200,200,600,350]);
XX = 1:length(vaild_y1);
plot(gca, XX(index_show), vaild_y1(index_show), yang_fu3_ku{1,yangsi_idnex(1)}, 'Color', color_list(yangsi_idnex(3),:), 'LineWidth', Line_Width(1))
hold(gca, 'on')
plot(gca, XX(index_show), y_vaild_predict1(index_show), yang_fu3_ku{1,yangsi_idnex(2)}, 'Color', color_list(yangsi_idnex(4),:), 'LineWidth', Line_Width(1), 'MarkerSize', makesize)
hold(gca, 'on')
title('Validation Set Testing Results')

set(gca, 'FontName', FontName, 'FontSize', FontSize, 'LineWidth', kuang_width)

xlabel(xlabel1)
ylabel(ylabel1)
legend(legend1)

box(gca, box1)
legend(gca, le_kuang) % Remove legend box
grid(gca, grid1)

%% 

for i = 1:6
    % Get prediction results and true values for each output dimension in training, validation, and test sets
    current_train_y = train_y(:, i);
    current_train_predict = y_train_predict(:, i);
    
    current_vaild_y = vaild_y(:, i);
    current_vaild_predict = y_vaild_predict(:, i);
    
    current_test_y = test_y(:, i);
    current_test_predict = y_test_predict(:, i);
    
    % Calculate training set metrics
    train_MAE = sum(abs(current_train_predict - current_train_y)) / length(current_train_y);
    train_MAPE = sum(abs((current_train_predict - current_train_y) ./ current_train_y)) / length(current_train_y);
    train_MSE = sum((current_train_predict - current_train_y).^2) / length(current_train_y);
    train_RMSE = sqrt(train_MSE);
    train_R2 = 1 - sum((current_train_y - current_train_predict).^2) / sum((current_train_y - mean(current_train_y)).^2);
    
    % Calculate validation set metrics
    vaild_MAE = sum(abs(current_vaild_predict - current_vaild_y)) / length(current_vaild_y);
    vaild_MAPE = sum(abs((current_vaild_predict - current_vaild_y) ./ current_vaild_y)) / length(current_vaild_y);
    vaild_MSE = sum((current_vaild_predict - current_vaild_y).^2) / length(current_vaild_y);
    vaild_RMSE = sqrt(vaild_MSE);
    vaild_R2 = 1 - sum((current_vaild_y - current_vaild_predict).^2) / sum((current_vaild_y - mean(current_vaild_y)).^2);
    
    % Calculate test set metrics
    test_MAE = sum(abs(current_test_predict - current_test_y)) / length(current_test_y);
    test_MAPE = sum(abs((current_test_predict - current_test_y) ./ current_test_y)) / length(current_test_y);
    test_MSE = sum((current_test_predict - current_test_y).^2) / length(current_test_y);
    test_RMSE = sqrt(test_MSE);
    test_R2 = 1 - sum((current_test_y - current_test_predict).^2) / sum((current_test_y - mean(current_test_y)).^2);
    
    % Print metrics for each output dimension
    disp(['--- Output Dimension ', num2str(i), ' ---']);
    
    % Print training set metrics
    disp(['Training Set:']);
    disp(['  MAE: ', num2str(train_MAE)]);
    disp(['  MAPE: ', num2str(train_MAPE)]);
    disp(['  MSE: ', num2str(train_MSE)]);
    disp(['  RMSE: ', num2str(train_RMSE)]);
    disp(['  R2: ', num2str(train_R2)]);
    
    % Print validation set metrics
    disp(['Validation Set:']);
    disp(['  MAE: ', num2str(vaild_MAE)]);
    disp(['  MAPE: ', num2str(vaild_MAPE)]);
    disp(['  MSE: ', num2str(vaild_MSE)]);
    disp(['  RMSE: ', num2str(vaild_RMSE)]);
    disp(['  R2: ', num2str(vaild_R2)]);
    
    % Print test set metrics
    disp(['Test Set:']);
    disp(['  MAE: ', num2str(test_MAE)]);
    disp(['  MAPE: ', num2str(test_MAPE)]);
    disp(['  MSE: ', num2str(test_MSE)]);
    disp(['  RMSE: ', num2str(test_RMSE)]);
    disp(['  R2: ', num2str(test_R2)]);
    
    disp('************************************************************************************');
end

%% 

analyzeNetwork(Mdl)

%% Time series probability interval prediction module

beta = G_out_data.beta;
eta = G_out_data.eta;
gailv_upper = []; % Structure to save upper and lower bounds
hidden_size = G_out_data.hidden_size; % Neural network neurons
label_vaild = ones(1, length(vaild_y_feature_label));
label_test = ones(1, length(test_y_feature_label));

Upper1 = []; Lower1 = [];
index_cluster_vaild = 1:length(label_vaild);
index_cluster_test = 1:length(label_test);

disp('************KDE Interval Prediction')

for N_dim = 1:size(y_test_predict,2)
    error_vaild = -(y_vaild_predict(:,N_dim) - vaild_y_feature_label(:,N_dim));
    
    kernel_label = 'normal';   % Kernel smoother type for kernel distribution 'normal' (default) 'box' 'triangle' 'epanechnikov'
    % Bandwidth default used by fitdist is optimal for estimating normal density
    
    pd_1 = fitdist(error_vaild, 'Kernel', 'Kernel', kernel_label); % Kernel density distribution
    [f, xi] = ksdensity(error_vaild, 'Bandwidth', pd_1.Bandwidth, 'Kernel', kernel_label);
    cdf = cumsum(f) / sum(f);
    disp(['Optimal bandwidth determined: ', num2str(pd_1.Bandwidth)])
    
    if N_dim == 1
        figure;
        set(gcf, 'color', 'w')
        h1 = histogram(error_vaild, 'Normalization', 'probability');
        hold on
        plot(xi, f, 'LineWidth', 1.2)
        hold on; grid on
        set(gcf, 'color', 'w')
        
        xlabel('Value'); ylabel('Probability')
        legend('Histogram', 'Probability Density Function')
        
        figure
        plot(xi, cdf, 'LineWidth', 1.2)
        hold on; grid on
        set(gcf, 'color', 'w')
        xlabel('Value'); ylabel('Probability')
        legend('Training Set Kernel Density Estimation')
    end
    
    for m = 1:length(beta)
        confidence_level = beta(m);
        upper_bound_get = xi(find(cdf >= 1-(1-confidence_level)/2, 1, 'first'));
        % Find upper and lower bounds of probability interval corresponding to confidence level
        lower_bound_get = xi(find(cdf >= (1-confidence_level)/2, 1, 'first'));
        
        C2_upper(:,m) = y_test_predict(:,N_dim) + upper_bound_get;
        C2_lower(:,m) = y_test_predict(:,N_dim) + lower_bound_get;
        
        Lower1 = C2_lower;
        Upper1 = C2_upper;
        Mdl1{N_dim,1}(:,m) = upper_bound_get;
        Mdl1{N_dim,2}(:,m) = lower_bound_get;
    end
    
    gailv_upper(N_dim).Upper1 = Upper1;
    gailv_upper(N_dim).Lower1 = Lower1;
end

Lower1_all = []; Upper1_all = []; test_y_feature_label_all = [];
for N_dim = 1:size(test_y_feature_label,2)
    Lower1_all = [Lower1_all; gailv_upper(N_dim).Lower1];
    Upper1_all = [Upper1_all; gailv_upper(N_dim).Upper1];
    test_y_feature_label_all = [test_y_feature_label_all; test_y_feature_label(:,N_dim)];
end

[value_result1] = interval_valuate1(Lower1_all, Upper1_all, test_y_feature_label_all, eta, beta);

value_lsit1 = [value_result1.PICP; value_result1.PINAW; value_result1.CWC; value_result1.MPICD; value_result1.AIS];

index1 = {'PICP','PINAW','CWC','MPICD','AIS'};
beta_str = [];
for j = 1:length(beta)
    beta_str{1,j} = num2str(beta(j));
end

value_lsit_table = array2table(value_lsit1);
value_lsit_table.Properties.VariableNames = (beta_str);
value_lsit_table.Properties.RowNames = index1;
disp(value_lsit_table)

show_wei_num = G_out_data.show_wei_num;
test_y_feature_label1 = test_y_feature_label;

test_y_feature_label = test_y_feature_label1(:, show_wei_num);

Lower1 = gailv_upper(:,show_wei_num).Lower1;
Upper1 = gailv_upper(:,show_wei_num).Upper1;

y_test_predict1 = y_test_predict(:, show_wei_num);

color_list1 = G_out_data.color_list1;
color_list2 = G_out_data.color_list2;
plot_index = G_out_data.plot_index;
yang_fu3_ku = G_out_data.yang_fu3_ku;
Line_Width = G_out_data.Line_Width;
yang_str3 = G_out_data.yang_str3;
FontName = G_out_data.FontName;
FontSize = G_out_data.FontSize;

kuang_width = G_out_data.kuang_width;
xlabel1 = G_out_data.xlabel1;
ylabel1 = G_out_data.ylabel1;
grid1 = G_out_data.grid1;
legend1 = G_out_data.legend1;
box1 = G_out_data.box1;
le_kuang = G_out_data.le_kuang;

yangsi_idnexqu = G_out_data.yangsi_idnexqu;
PlotProbability1(y_test_predict1, test_y_feature_label, Lower1, Upper1, plot_index(1), plot_index(2), 'Data', color_list2(yangsi_idnexqu(1),:), color_list2(yangsi_idnexqu(2),:), color_list2(yangsi_idnexqu(3),:), 'Interval Prediction', beta);  % Background color/interval color/true value color/prediction color

Upper_pre1 = [];  % Upper bound for formal prediction
Lower1_pre1 = []; % Lower bound for formal prediction

for N_dim = 1:size(y_test_predict,2)
    for m = 1:length(beta)
        upper_bound_get = Mdl1{N_dim,1}(:,m);
        lower_bound_get = Mdl1{N_dim,2}(:,m);
        
        Upper_pre1(N_dim,m) = y_last_predict1(N_dim) + upper_bound_get;
        Lower1_pre1(N_dim,m) = y_last_predict1(N_dim) + lower_bound_get;
    end
    
    gailv_upper(1).Upper_pre1 = Upper_pre1;
    gailv_upper(1).Lower1_pre1 = Lower1_pre1;
end

figure('Units', 'pixels', 'Position', [100 100 700 475]); % Initialize display position
if length(y_before) > length(y_last_predict1)*5
    y_before = y_before(end-length(y_last_predict1)*5+1:end);
end

x = length(y_before)+1:length(y_before)+length(y_last_predict1);

for j = 1:size(Lower1_pre1,2)
    b1(j) = fill([x, fliplr(x)], [Lower1_pre1(:,j)', fliplr(Upper_pre1(:,j)')], color_list2(yangsi_idnexqu(1),:), 'FaceAlpha', 0.1+1-beta(j));
    b1(j).EdgeColor = 'none';
    hold on
    h1 = plot(gca, 1:length(y_before), y_before, yang_fu3_ku{1,yangsi_idnexqu(3)}, 'Color', color_list2(yangsi_idnexqu(2),:), 'LineWidth', Line_Width(1));
    hold on
end

set(gca, 'FontName', FontName, 'FontSize', FontSize, 'LineWidth', kuang_width)

xlabel(xlabel1)
ylabel(ylabel1)
str1 = [];
for n = 1:length(beta)
    str1{1,n} = [num2str(100*beta(n)), '% confidence interval prediction'];
end
str1{1,length(beta)+1} = 'Historical data';

legend([b1,h1], str1, 'NumColumns', 3)

box(gca, box1)
legend(gca, le_kuang) % Remove legend box
grid(gca, grid1)