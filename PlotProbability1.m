function PlotProbability1(Predict, True_test, Lower, Upper, llimit, rlimit, Name, IColor, RColor, TColor, title_str, beta)
  
    figure('Position', [100, 100, 1000, 600]); 
    transparence = [0.5; 0.5; 0.7; 0.7]; %
    x = 1:length(Predict);
    h = gca;
    FColor = [1, 1, 1]; 
    hp = patch([x(1), x(end), x(end), x(1)], ...
    [min([min(Lower), min(True_test)]), min([min(Lower), min(True_test)]), max([max(Upper), max(True_test)]), max([max(Upper), max(True_test)])], FColor, 'FaceVertexAlphaData', transparence, 'FaceAlpha', "interp");
    uistack(hp, 'bottom');
    hold on
    % n = [0.25; 0.3; 0.35; 0.4; 0.45; 0.5; 0.6]; 
    for j = 1:size(Lower, 2)
        window(j) = fill([x, fliplr(x)], [Lower(:, j)', fliplr(Upper(:, j)')], IColor, 'FaceAlpha', 0.1 + (1 - beta(j))); %, 'DisplayName', [num2str(100*beta(j)), '% confidence interval']
        window(j).EdgeColor = 'none';
        hold on
        plot(Upper(:, j), 'Marker', "none", "LineStyle", "none", "Tag", 'none', "Visible", "off");
        hold on
        plot(Lower(:, j), 'Marker', "none", "LineStyle", "none", "Tag", 'none', "Visible", "off");
        hold on
    end
    h1 = plot(True_test, '*', 'MarkerSize', 4, 'Color', RColor, 'DisplayName', 'True value');
    hold on
    h2 = plot(Predict, 'Color', TColor, 'LineWidth', 1.5, 'DisplayName', 'Predicted value');
    hold on
    xlim([llimit rlimit]);
    str1 = [];
    for n = 1:length(beta)
        str1{1, n} = [num2str(100 * beta(n)), '% confidence interval'];
    end
    str1{1, length(beta) + 1} = 'True value';
    str1{1, length(beta) + 2} = 'Predicted value';
 
    title(title_str)
    xlabel('Sample point', "FontSize", 12, "FontWeight", "bold");
    ylabel('Predicted value', "FontSize", 12, "FontWeight", "bold");
    legend('Location', 'bestoutside')
   
    legend([window, h1, h2], str1, 'Location', 'bestoutside')
    
    grid on
    box off

end