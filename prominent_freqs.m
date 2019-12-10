% Performs fourier analysis and determines the 3 most prominent %
% frequencies of basketball dribbling data %
function [] = prominent_freqs(readfile, writefile)
    acceleration = load(readfile).Acceleration;
    
    % Convert time to seconds
    d_t = posixtime(acceleration.Timestamp);
    d_t = d_t - d_t(1);
    
    % Perform fourier analysis
    [x_freq, x_amp] = dft(d_t, acceleration.X);
    [z_freq, z_amp] = dft(d_t, acceleration.Z);
        
    % Get n most prominent frequencies
    [x_freq, x_amp] = get_peaks(x_freq, x_amp, 3);
    [z_freq, z_amp] = get_peaks(z_freq, z_amp, 3);
        
    % Build struct
    s = struct('x_freq', x_freq, 'x_amp', x_amp, ...
               'z_freq', z_freq, 'z_amp', z_amp);
    
    % Write to disk
    save(writefile, 's');
end

function [freq, amp] = dft(t, a)
    % Apply a 7 point moving average filter
    aFiltered = movmean(a, 7);
    
    N = length(aFiltered);
    
    % Calculate sampling frequency
    Fs = 1/(t(2)-t(1));
    
    % 0 centered shifted range of frequencies
    freq = linspace(-Fs/2 , Fs/2 - Fs/N, N) + Fs/(2*N)*mod(N,2);
    
    % Calculate DFT of mean subtracted acceleration
    % Remove 0 frequency (artifact of FFT)
    amp = abs(fftshift(fft(aFiltered - mean(aFiltered))));
end

function [freq, amp] = get_peaks(f, a, n)
    % Find frequency peaks
    [peaks, p_i] = findpeaks(a);
    % n most prominent frequencies
    halfway = ceil(length(p_i) / 2);
    [max, max_i] = maxk(peaks(1:halfway), n);
    freq = [];
    amp = [];
    for i = 1:n
        freq = [freq -f(p_i(max_i(i)))];
        amp = [amp max(i)];
    end
end


