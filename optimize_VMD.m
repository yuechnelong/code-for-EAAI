function [Mdl, Convergence_curve, fMin, pos] = optimize_VMD(data_w, num_pop, num_iter, method_mti)

    
    pop = num_pop;
    M = num_iter;
    LB = [2, 15];
    UB = [9, 100];
    nvars = length(LB);
    fit_ceemdan1 = @fit_ceemdan;
    
    if strcmp(method_mti, 'GRIME') == 1
        [fMin, Mdl, Convergence_curve, pos] = GRIME_GRIME(pop, M, LB, UB, nvars, fit_ceemdan1, data_w);
    else
        error('1');
    end
    
  
    figure
    plot(Convergence_curve, '--p', 'LineWidth', 1.2)
    xticks([1:length(Convergence_curve)])
    title('optimize process')
    xlabel('iter')
    ylabel('fitness')
    grid on
    set(gca, "FontName", "Times New Roman", "FontSize", 12, "LineWidth", 1.2)
    box off
    
    disp([method_mti, 'VMD:   ', "  deo_num:", num2str(round(pos(1))), "  maxiter:", num2str(round(pos(2)))])
end

function [fitness_value, Mdl] = fit_ceemdan(pop, data_w)

    
    deo_num = round(pop(1));
    MaxIter = round(pop(2));
    
    
    [imf, res] = vmd(data_w, 'NumIMF', deo_num, 'MaxIterations', MaxIter);
    u = [imf, res]';
    
  
    M = size(u, 1);
    for i = 1:M
        data1 = abs(hilbert(u(i, :))); 
        data2 = data1 / sum(data1);
        s_value = 0;
        for ii = 1:size(data2, 2)
            if data2(1, ii) > 0  
                value = data2(1, ii) * log(data2(1, ii));
                s_value = s_value + value;
            end
        end
        fitness(i, :) = -s_value;
    end
    
    [fitness_value] = min(fitness);
    Mdl = u;
end

function [Best_rime_rate, Mdl_best, Convergence_curve, Best_rime] = GRIME_GRIME(N, Max_iter, lb, ub, dim, fobj, data_w)

    
    Best_rime = zeros(1, dim);
    Best_rime_rate = inf; %
    Convergence_curve = zeros(1, Max_iter);
    
   
    label = 5; 
    Rimepop = yinshe(N, dim, label, ub, lb);
    Rimepop = min(ub, Rimepop); % bound violating to upper bound
    Rimepop = max(lb, Rimepop); % bound violating to lower bound
    
    Lb = lb .* ones(1, dim); % lower boundary 
    Ub = ub .* ones(1, dim); % upper boundary
    it = 1; % Number of iterations
    Rime_rates = zeros(1, N); % Initialize the fitness value
    newRime_rates = zeros(1, N);
    W = 5; % Soft-rime parameters
    
    % Calculate the fitness value of the initial position
    for i = 1:N
        [Rime_rates(1, i), Mdl_all{1, i}] = fobj(Rimepop(i, :), data_w);
        % Make greedy selections
        if Rime_rates(1, i) < Best_rime_rate
            Best_rime_rate = Rime_rates(1, i);
            Best_rime = Rimepop(i, :);
            Mdl_best = Mdl_all{1, i};
        end
    end
    
    % Main loop
    while it <= Max_iter
        RimeFactor = (rand-0.5)*2*cos((pi*it/(Max_iter/10)))*(1-round(it*W/Max_iter)/W);
        E = sqrt(it/Max_iter);
        newRimepop = Rimepop; % Recording new populations
        normalized_rime_rates = normr(Rime_rates);
        
        for i = 1:N
            for j = 1:dim
                % Soft-rime search strategy
                r1 = rand();
                if r1 < E
                    newRimepop(i, j) = Best_rime(1, j) + RimeFactor * ((Ub(j)-Lb(j))*rand + Lb(j));
                end
                % Hard-rime puncture mechanism
                r2 = rand();
                if r2 < normalized_rime_rates(i)
                    newRimepop(i, j) = Best_rime(1, j);
                end
            end
        end
        
        for i = 1:N
            % Boundary absorption
            Flag4ub = newRimepop(i, :) > ub;
            Flag4lb = newRimepop(i, :) < lb;
            newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;
            
            [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), data_w);
            
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
        
       
        for i = 1:N
            k = (1 + (it/Max_iter)^0.5)^10;
            newRimepop(i, :) = (ub+lb)/2 + (ub+lb)/(2*k) - Rimepop(i, :)/k;
            Flag4ub = newRimepop(i, :) > ub;
            Flag4lb = newRimepop(i, :) < lb;
            newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;
            
            [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), data_w);
            
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

function result = yinshe(N, dim, label, ub, lb)

    if label == 1
        % tent 
        tent = 1.2;
        Tent = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                if Tent(i, j-1) < tent
                    Tent(i, j) = Tent(i, j-1)/tent;
                elseif Tent(i, j-1) >= tent
                    Tent(i, j) = (1-Tent(i, j-1))/(1-tent);
                end
            end
        end
        result = lb + Tent .* (ub-lb);
    elseif label == 2
        % chebyshev 
        chebyshev = 2;
        Chebyshev = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Chebyshev(i, j) = cos(chebyshev .* acos(Chebyshev(i, j-1)));
            end
        end
        result = lb + (Chebyshev+1)/2 .* (ub-lb);
    elseif label == 3
        % singer 
        u = 1;
        singer = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                singer(i, j) = u*(7.86*singer(i, j-1) - 23.31*singer(i, j-1).^2 + ...
                                28.75*singer(i, j-1).^3 - 13.302875*singer(i, j-1).^4);
            end
        end
        result = lb + singer .* (ub-lb);
    elseif label == 4
        % Logistic ”≥…‰
        miu = 2;
        Logistic = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Logistic(i, j) = miu .* Logistic(i, j-1) .* (1-Logistic(i, j-1));
            end
        end
        result = lb + Logistic .* (ub-lb);
    elseif label == 5
        % Sine ”≥…‰
        sine = 2;
        Sine = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Sine(i, j) = (4/sine) * sin(pi*Sine(i, j-1));
            end
        end
        result = lb + Sine .* (ub-lb);
    elseif label == 6
    
        a = 0.5; b = 0.6;
        Circle = rand(N, dim);
        for i = 1:N
            for j = 2:dim
                Circle(i, j) = mod(Circle(i, j-1) + a - b/(2*pi)*sin(2*pi*Circle(i, j-1)), 1);
            end
        end
        result = lb + Circle .* (ub-lb);
    else
        % Œﬁ”≥…‰
        result = lb + rand(N, dim) .* (ub-lb);
    end
end