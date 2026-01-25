function [pos_get, Mdl, fMin, Convergence_curve] = optimize_fitrfeature_selected1(A_data1, predict_num, num_pop, num_iter, method_mti)
  
pop = num_pop;
M = num_iter;
LB = zeros(1, size(A_data1, 2) - predict_num);
UB = ones(1, size(A_data1, 2) - predict_num);
nvars = length(LB);
fit_fitrensemble1 = @fit_fitrtree;

if strcmp(method_mti, 'GRIME') == 1
    [fMin, Mdl, Convergence_curve, pos] = GRIME_GRIME(pop, M, LB, UB, nvars, fit_fitrensemble1, A_data1, predict_num);
else
    error('only');
end

pos_get = find(round(pos) == 1); 
if isempty(pos_get)
    pos_get(1) = 1;
end

disp('******************************');
disp('Optimization iteration value of the feature selection objective function：');
disp(Convergence_curve);    
disp([method_mti, 'Optimize feature selection:   ', "Number of selected features:", num2str(length(pos_get))]);
end

%% 
function [fitness_value, Mdl] = fit_fitrtree(pop, A_data1, predict_num)
pop_get = round(pop);
pop_get_index = find(pop_get == 1); 
if isempty(pop_get_index)
    pop_get(1) = 1;
end
num_train_get = round(0.8 * size(A_data1, 1));
random_label = randperm(size(A_data1, 1));

Mdl = TreeBagger(20, A_data1(random_label(1:num_train_get), (pop_get == 1)), ...
                A_data1(random_label(1:num_train_get), end-predict_num+1), 'Method', 'regression');
    
P_vaild_y_feature_label = predict(Mdl, A_data1(random_label(num_train_get:end), (pop_get == 1)));
fitness_value = sum(sum(abs(P_vaild_y_feature_label - A_data1(random_label(num_train_get:end), end)))) / length(P_vaild_y_feature_label);
end

%% 
function [Best_rime_rate, Mdl_best, Convergence_curve, Best_rime] = GRIME_GRIME(N, Max_iter, lb, ub, dim, fobj, A_data1, predict_num)

Best_rime = zeros(1, dim);
Best_rime_rate = inf; 
Convergence_curve = zeros(1, Max_iter);
label = 5; % 
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
    [Rime_rates(1, i), Mdl_all{1, i}] = fobj(Rimepop(i, :), A_data1, predict_num);
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
        
        [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), A_data1, predict_num);
        
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
        
        [newRime_rates(1, i), Mdl_new{1, i}] = fobj(newRimepop(i, :), A_data1, predict_num);
        
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

%% 
function result = yinshe(N, dim, label, ub, lb)

if label == 1
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
    chebyshev = 2;
    Chebyshev = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            Chebyshev(i, j) = cos(chebyshev .* acos(Chebyshev(i, j-1)));
        end
    end
    result = lb + (Chebyshev+1)/2 .* (ub-lb);
elseif label == 3
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
    miu = 2;
    Logistic = rand(N, dim);
    for i = 1:N
        for j = 2:dim
            Logistic(i, j) = miu .* Logistic(i, j-1) .* (1-Logistic(i, j-1));
        end
    end
    result = lb + Logistic .* (ub-lb);
elseif label == 5
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
    result = lb + rand(N, dim) .* (ub-lb);
end
end