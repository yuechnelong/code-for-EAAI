function [] = plotpl(data, IMF)
%   绘制imf - Plot IMFs
%   此处显示详细说明 - Detailed description here
num = size(IMF, 1);
for k_first = 0:num:size(IMF,1)-1
    clear num_k;
    figure('Name', 'Decomposition Plot', 'Color', [1 1 1])
    for num_k = 1:min(num, size(IMF,1)-k_first)
        subplot(num, 1, num_k);
        plot(IMF(k_first+num_k, :)); axis('tight');
        if (k_first == 0 && num_k == 1)
%             title('CEEMDAN');
        end
       if k_first + num_k < size(IMF, 1)
        ylabel(['IMF', num2str(k_first+num_k)]);
       else
        ylabel('res');
        end
    end
end
set(gcf, 'color', 'w')

fs = 1;
N = length(data);
s = data(1:N);
t = (0:N-1)/fs;
u = IMF;
figure('Name', 'Spectrum Plot', 'Color', [1 1 1])
CC_mat = [];
for i = 1:size(u, 1) 
    subplot(size(u, 1), 1, i)
    [cc, y_f] = FFT_plot(u(i,:), fs, 1);
    plot(y_f, cc, 'b', 'LineWIdth', 1);
    ylabel(['IMF', num2str(i)]);
    CC_mat(i).x = y_f;    % X-axis of spectrum plot
    CC_mat(i).y = cc;      % Y-axis of spectrum plot
end
save pingpu CC_mat
disp('Frequency spectrum data for each component saved in pingpu.mat')
end
%%
function [cc, y_f] = FFT_plot(y, fs, style, varargin)

nfft = 2^nextpow2(length(y)); % Find the largest power of 2 greater than the length of y (automatically calculates optimal FFT step size nfft)
% nfft = 1024; % Manually set FFT step size nfft
y = y - mean(y); % Remove DC component
y_ft = fft(y, nfft); % Perform DFT on signal y to obtain frequency amplitude distribution
y_p = y_ft .* conj(y_ft) / nfft; % conj() function calculates the complex conjugate of y, the conjugate of a real number is itself
y_f = fs * (0:nfft/2-1) / nfft; % Frequency sequence corresponding to FFT transform
% y_p = y_ft .* conj(y_ft) / nfft; % conj() function calculates the complex conjugate of y, the conjugate of a real number is itself
if style == 1
    if nargin == 3
        cc = 2 * abs(y_ft(1:nfft/2)) / length(y);
        % ylabel('Amplitude'); xlabel('Frequency'); title('Signal Amplitude Spectrum');
    else
        f1 = varargin{1};
        fn = varargin{2};
        ni = round(f1 * nfft/fs + 1);
        na = round(fn * nfft/fs + 1);
        hold on
        plot(y_f(ni:na), abs(y_ft(ni:na)*2/nfft), 'k');
    end
elseif style == 2
    plot(y_f, y_p(1:nfft/2), 'k');
    % ylabel('Power Spectral Density'); xlabel('Frequency'); title('Signal Power Spectrum');
else
    subplot(211); plot(y_f, 2*abs(y_ft(1:nfft/2))/length(y), 'k');
    ylabel('Amplitude'); xlabel('Frequency'); title('Signal Amplitude Spectrum');
    subplot(212); plot(y_f, y_p(1:nfft/2), 'k');
    ylabel('Power Spectral Density'); xlabel('Frequency'); title('Signal Power Spectrum');
end
end