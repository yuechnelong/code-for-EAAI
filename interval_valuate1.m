function [value_result] = interval_valuate1(Lower, Upper, Real, eta, beta)
   % Evaluate performance metrics for interval prediction
   % PICP metric - Proportion of true values falling within upper and lower bounds - larger is better
   % mean_PICP metric - Average under different confidence levels
   [PICP, mean_PICP] = PICP_FUN(Lower, Upper, Real);
   value_result.PICP = PICP;
   value_result.mean_PICP = mean_PICP;
   disp(PICP)
   
   % PINAW metric - Narrowness of error band - smaller is better | Narrower PI intervals are considered more informative than wider ones
   % mean_PINAW metric - Average under different confidence levels
   [PINAW, mean_PINAW] = PINAW_FUN(Lower, Upper, Real);
   value_result.PINAW = PINAW;
   value_result.mean_PINAW = mean_PINAW;
   
   % CWC - Comprehensive consideration of coverage and narrowness, smaller is better, some intelligent optimization algorithms also use this as an objective function
   % Eta is the coefficient of the penalty function, larger values indicate greater penalty for not reaching the confidence level
   [CWC, mean_CWC] = CWC_FUN(PINAW, PICP, eta, beta);
   value_result.CWC = CWC;
   value_result.mean_CWC = mean_CWC;
   
   % MPICD - Considers the middle of the error band and the true value, smaller is better
   [MPICD, mean_MPICD] = MPICDF(Lower, Upper, Real);
   value_result.MPICD = MPICD;
   value_result.mean_MPICD = mean_MPICD;
   
   % AIS interval quantile - Comprehensive consideration of coverage and interval width, smaller is better
   [AIS, mean_AIS] = AIS_FUN(Lower, Upper, Real, beta);
   value_result.AIS = AIS;
   value_result.mean_AIS = mean_AIS;
end

function [PICP, mean_PICP] = PICP_FUN(Lower, Upper, Real)
    temp = zeros(size(Lower, 2), 1);
    for i = 1:size(Lower, 2)
        for j = 1:length(Real)
            if Lower(j, i) <= Real(j) && Upper(j, i) >= Real(j)
                temp(i, :) = temp(i, :) + 1;
                count_picp(:, i) = temp(i, :);
            else
                temp(i, :) = temp(i, :) + 0;
                count_picp(:, i) = temp(i, :);
            end
        end
    end
    PICP = count_picp / length(Real);
    mean_PICP = mean(PICP);
end

function [PINAW, mean_PINAW] = PINAW_FUN(Lower, Upper, Real)
    PINAW = sum(Upper - Lower) / (length(Real) * (max(Real) - min(Real)));
    mean_PINAW = mean(PINAW);
end

function [CWC, mean_CWC] = CWC_FUN(PINAW, PICP, Eta, Beta)
   % CWC - Coverage Width Criterion
    PINAW = PINAW';
    PICP = PICP';
    for i = 1:numel(PINAW)
        if PICP(i) < Beta(i)
            Gamma(i) = 1;
        else
            Gamma(i) = 0;
        end
    end
    for m = 1:numel(PINAW)
        CWC(m) = PINAW(m) * (1 + Gamma(m) * exp(-Eta * (PICP(m) - Beta(m))));
    end
    mean_CWC = mean(CWC);
end

function [MPICD, mean_MPICD] = MPICDF(Lower, Upper, Real)
    MPICD = sum(abs(Upper + Lower - 2 * Real)) / 2 / (length(Real));
    mean_MPICD = mean(MPICD);
end

function [AIS, mean_AIS] = AIS_FUN(Lower, Upper, Real, beta)
    S = [];
    for i = 1:size(Lower, 2)
        for j = 1:length(Real)
            theta = Upper(j, i) - Lower(j, i); % Predicted width
            betai = 1 - beta(i); % i-th confidence interval
            
            if Lower(j, i) <= Real(j) && Upper(j, i) >= Real(j)
                S(i, j) = -2 * betai * theta;
            elseif Lower(j, i) > Real(j)
                S(i, j) = -2 * betai * theta - 4 * (Lower(j, i) - Real(j));
            elseif Upper(j, i) < Real(j)
                S(i, j) = -2 * betai * theta - 4 * (Real(j) - Upper(j, i));
            end
        end
    end
    AIS = abs(mean(S'));
    mean_AIS = mean(AIS);
end