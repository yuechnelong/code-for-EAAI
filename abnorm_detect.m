function [y, index_ab_unique] = abnorm_detect(x1)
% Segmentation + Isolation Forest combined anomaly detection method
rng(1)

% Outlier/missing value handling
% Mainly for missing value interpolation processing
data1 = fillmissing(x1, 'linear'); % Linear interpolation
% Outlier handling
% Identify outliers and correct them through interpolation, outliers defined as values deviating from the mean by more than three times the standard deviation
[B_data, TF, L, U, C] = filloutliers(data1(:, :), "clip", "movmedian", 15); % Sliding window linear interpolation
data_new = B_data;

figure
index_set = 1:ceil(size(data1, 1) / 10);
plot(data1(index_set, end))
hold on
plot(B_data(index_set, end), "o-")
hold on
legend("Original Data", "Filled Data")
title('Data Preprocessing Comparison')
% Find the most relevant data column / perform data discrimination
data_new = zscore(data_new);

cor_R2 = corrcoef(data_new);
cor_R2_end = cor_R2(end, :);
cor_R2_end(end) = [];
[~, index_max] = max(cor_R2_end);

% Partition each segment

index_abnorm1 = find(data_new(:, index_max) > (mean(data_new(:, index_max)) + 3 * std(data_new(:, index_max))));
index_abnorm2 = find(data_new(:, index_max) < (mean(data_new(:, index_max)) - 3 * std(data_new(:, index_max))));
index_abnorm_all1 = [index_abnorm1; index_abnorm2];
data_new(index_abnorm_all1, :) = [];
fitPoints = [data_new(:, index_max) data_new(:, end)]; % Data fitting for the most relevant factor and output y

nbin = 10; % Divide X into 10 segments

xpoint_bin = linspace(min(data_new(:, index_max)), max(data_new(:, index_max)), 10);

fitFcn = polyfit(fitPoints(:, 1), fitPoints(:, 2), 2); % Perform quadratic fitting, can also be adjusted according to actual situation

ypoint_bin = polyval(fitFcn, xpoint_bin);

figure
scatter(fitPoints(:, 1), fitPoints(:, 2))
hold on
plot(xpoint_bin, ypoint_bin, '--')
xlabel(sprintf('Feature %d (Most Relevant)', index_max))
ylabel('Target Variable')
title('Data Fitting and Segmentation')
% Perform isolation forest anomaly recognition for multiple interval segments separately, using anomaly points for boundary drawing
ContaminationFraction_rate = 0.05;
if size(data_new, 1) > 50000
    ContaminationFraction_rate = 0.02;
elseif size(data_new, 1) > 80000
    ContaminationFraction_rate = 0.01;
end

norm_index_all_upper = [];
norm_index_all_lower = [];
norm_index_all = [];
abnorm_index_all = [];
for i = 1:nbin-1
% for i = 1
    index = find((fitPoints(:, 1) > xpoint_bin(i)) & (fitPoints(:, 1) < xpoint_bin(i + 1))); % Extract data for this segment
    if (length(index) > size(x1, 1) / nbin)
        ContaminationFraction_rate1 = ContaminationFraction_rate / ((length(index) / size(x1, 1)) / (1 / nbin));
    else
        ContaminationFraction_rate1 = ContaminationFraction_rate;
    end

    y_data = fitPoints(index, end);
    mean_line_data = (ypoint_bin(i) + ypoint_bin(i + 1)) / 2;
    [forest, tf_forest, s_forest] = iforest(y_data, ContaminationFraction = ContaminationFraction_rate1); % Set 0.02 data partition ratio
    index1 = index;
    index_abnorm = index1((tf_forest == 1));

    data_index_abnorm = fitPoints(index_abnorm, end);

    index_abnorm_upper = index_abnorm(data_index_abnorm > mean_line_data);
    index_abnorm_lower = index_abnorm(data_index_abnorm < mean_line_data);

    data_index_abnorm_up = fitPoints(index_abnorm_upper, end);

    index_abnorm_upper1 = index_abnorm_upper(data_index_abnorm_up > prctile(data_index_abnorm_up, 50));

    index_norm_upper1 = index_abnorm_upper(data_index_abnorm_up < prctile(data_index_abnorm_up, 50));

    index_abnorm = [index_abnorm_lower; index_abnorm_upper1];

    index1((tf_forest == 1)) = [];

    abnorm_index_all = [abnorm_index_all; index_abnorm];

    index_norm = [index1; index_norm_upper1];

    norm_index_all = [norm_index_all; index_norm];

    index_norm_upper = index1(fitPoints(index1, end) > prctile(fitPoints(index1, end), 99));
    index_norm_lower = index1(fitPoints(index1, end) < prctile(fitPoints(index1, end), 2));

    norm_index_all_upper = [norm_index_all_upper; index_norm_upper];
    norm_index_all_lower = [norm_index_all_lower; index_norm_lower];
    % abnorm_cell{1, i} = find();
end
% Find missing data and normal data



norm_index_up_data = [fitPoints(norm_index_all_upper, 1) fitPoints(norm_index_all_upper, end)];

norm_index_lower_data = [fitPoints(norm_index_all_lower, 1) fitPoints(norm_index_all_lower, end)];

fitFcn_low = polyfit(norm_index_lower_data(:, 1), norm_index_lower_data(:, 2), 2); % Perform quadratic fitting, can also be adjusted according to actual situation

ypoint_bin_low = polyval(fitFcn_low, xpoint_bin);

fitFcn_up = polyfit(norm_index_up_data(:, 1), norm_index_up_data(:, 2), 2); % Perform quadratic fitting, can also be adjusted according to actual situation

ypoint_bin_up = polyval(fitFcn_up, xpoint_bin);


%
input_norm_data = B_data;
input_norm_data(index_abnorm_all1, :) = [];
% input_norm_data(abnorm_index_all, :) = [];
index_set = 1:size(fitPoints, 1);
% index_set(abnorm_index_all1) = [];

input_norm_data1 = input_norm_data;

x_label_get = fitPoints(:, 1);
y_label_get = fitPoints(:, 2);

y_label_get_lower = polyval(fitFcn_low, x_label_get);
y_label_get_upper = polyval(fitFcn_up, x_label_get);

index_up_ubnorm = find(y_label_get > y_label_get_upper + 0.3);
index_low_ubnorm = find(y_label_get < y_label_get_lower - 0.2);
data_ubnorm2 = [index_up_ubnorm; index_low_ubnorm];
abnorm_index = index_set(data_ubnorm2);

abnorm_index_all1 = unique([abnorm_index']);
abnorm_index_data = [fitPoints(abnorm_index_all1, 1) fitPoints(abnorm_index_all1, end)];

input_norm_data1(abnorm_index_all1, :) = [];
y = input_norm_data1;

figure
scatter(fitPoints(:, 1), fitPoints(:, 2), 'Color', [0.2353 0.5176 0.7725])
hold on
plot(xpoint_bin, ypoint_bin, '--', 'LineWidth', 1.5)
hold on
scatter(abnorm_index_data(:, 1), abnorm_index_data(:, 2), 'Color', [0.9725 0.6039 0.1922])
hold on
% scatter(input_norm_data(data_ubnorm2, index_max), input_norm_data(data_ubnorm2, end), 'Color', [0.9725 0.6039 0.1922])
% hold on
plot(xpoint_bin, ypoint_bin_low, '-', 'LineWidth', 1.5, 'Color', [0.9725 0.6039 0.1922])
hold on
plot(xpoint_bin, ypoint_bin_up, '-', 'LineWidth', 1.5, 'Color', [0.9725 0.6039 0.1922])
xlabel(sprintf('Feature %d (Most Relevant)', index_max))
ylabel('Target Variable')
legend('Normal value', 'data fit line', 'Anomaly value', 'Lower bound', 'Upper bound')
title('Final Anomaly Detection Results')

disp("Rows with anomalous values:")
disp(abnorm_index_all1')
index_ab_unique = abnorm_index_all1';

end