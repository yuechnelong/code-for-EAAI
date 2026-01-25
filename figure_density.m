function [] = figure_density(color_index, y_test_predict, test_y, str)
    % Calculate kernel density function
    figure
    ke = ksdensity([test_y, y_test_predict], [test_y, y_test_predict]);

    % Plot scatter points
    markerSize = 15;
    h1 = scatter(test_y, y_test_predict, markerSize, ke, 'filled');
    ytick = linspace(min(y_test_predict), max(y_test_predict), 5);
    % Color settings
    % colormap(jet);
    % colormap(slanCM(98));
    
    colormap(slanCM(color_index));

    caxis([min(ke), max(ke)])
    cbtick = linspace(min(ke), max(ke), 3);
    colorbar('Ticks', cbtick, 'TickLabels', {'L', 'M', 'H'});
    hold on; box on

    % Draw a fitting line
    pcoef = polyfit(test_y, y_test_predict, 1);  % 1D curve fitting
    yfited = polyval(pcoef, test_y);

    h2 = plot(test_y, yfited, 'Color', [0.2588 0.2941 0.3137] * 1.5, 'LineWidth', 1.5);
    xlim([min(test_y) max(test_y)]); ylim([min(y_test_predict) max(y_test_predict)]);
    Title1 = title('x-y fit');
    xlabel('True data');
    ylabel('Predicted data');

    lm = fitlm(test_y, y_test_predict);
    % Get R^2 coefficient
    R2 = lm.Rsquared.Ordinary; % R-squared coefficient

    a1 = num2str(sprintf('%0.3f', pcoef(1)));
    a2 = num2str(sprintf('%0.3f', pcoef(2)));
    txt_set = ['Num = ', num2str(length(test_y)) ...
        newline 'y = ', a1, 'x + ', a2 ...
        newline 'R^2 = ', num2str(R2)];

    % Display text position
    text(min(test_y) + 0.05 * (max(test_y) - min(test_y)), max(y_test_predict) - 0.1 * (max(y_test_predict) - min(y_test_predict)), txt_set, 'color', 'k', 'FontName', 'Times New Roman', 'FontSize', 10)
    
    % Title
    title([str, ' density plot'])
    
    % Legend
    legend([h1, h2], {'True-predict data', 'Fit data'}, ...
        'Location', 'SouthEast', ...
        'Box', 'on');

    % Coordinate axes settings
    set(gca, 'LineWidth', 1, ...                                 % Line width
        'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
        'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
        'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
        'xColor', [.1 .1 .1], 'YColor', [.2 .2 .2])      % Axis color

    % Font and size
    % set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
    set(gca, 'FontSize', 11, 'LineWidth', 1.2)
    % set(Title1, 'FontSize', 12, 'FontWeight', 'bold')
end

function colorList = slanCM(type, num)
% @author : slandarer
% -----------------------
% type : type of colorbar
% num  : number of colors
if nargin < 2
    num = 256;
end
if nargin < 1
    type = '';
end

slanCM_Data = load('map_Color_Data.mat');
CList_Data = [slanCM_Data.slandarerCM(:).Colors];
% disp(slanCM_Data.author);

if isnumeric(type)
    Cmap = CList_Data{type};
else
    Cpos = strcmpi(type, slanCM_Data.fullNames);
    Cmap = CList_Data{Cpos};
end

Ci = 1:256; Cq = linspace(1, 256, num);
colorList = [interp1(Ci, Cmap(:, 1), Cq, 'linear')', ...
             interp1(Ci, Cmap(:, 2), Cq, 'linear')', ...
             interp1(Ci, Cmap(:, 3), Cq, 'linear')'];
end