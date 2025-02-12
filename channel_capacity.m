% DCMC capacity for M-ary modulations on AWGN or Rayleigh flat channel
clearvars; clc;
%---------Input Fields------------------------
nSym = 10^4; % Number of symbols to transmit
channelModel = 'AWGN'; % Channel model - 'AWGN' or 'RAYLEIGH'
snrdB = -10:0.1:30; % SNRs in dB for noise generation
MOD_TYPE = 'PSK'; % Set 'PSK' or 'QAM' or 'PAM'
arrayOfM = [2, 4, 8, 16]; % Array of M values to simulate
% arrayOfM=[4,16,64,256]; %array of M values for MOD_TYPE='QAM'

plotColor = ['b', 'g', 'c', 'm', 'k']; j = 1; % Plot colors/color index
legendString = cell(1, length(arrayOfM)); % For legend entries
for M = arrayOfM
    C = zeros(1, length(snrdB)); % Capacity

    d = ceil(M.*rand(1, nSym)); % Uniformly distributed source syms
    [s, constellation] = modulate(MOD_TYPE, M, d); % Constellation mapping

    for i = 1:length(snrdB)
        if strcmpi(channelModel, 'RAYLEIGH') % Rayleigh flat channel
            h = 1/sqrt(2)*(randn(1, nSym) + 1i*randn(1, nSym));
        else % Else assume no channel effect
            h = ones(1, nSym);
        end
        hs = h.*s; % Channel effect on the modulated symbols
        [r, ~, N0] = add_awgn_noise(hs, snrdB(i)); % r = h*s+n (received)

        % Calculate conditional probabilities of each const. point
        pdfs = exp(-(abs(ones(M, 1)*r - constellation.'*h).^2)/(2*N0));
        p = max(pdfs, realmin); % Prob of each constellation points
        p = p ./ (ones(M, 1) * sum(p)); % Normalize probabilities
        symEntropy = -sum(p.*log2(p)); % Senders uncertainty
        C(i) = log2(M) - mean(symEntropy); % Bits/sym-senders uncertainty
    end
    plot(snrdB, C, 'LineWidth', 1.0, 'Color', plotColor(j)); hold on;
    legendString{j} = [num2str(M), '-', MOD_TYPE]; j = j + 1;
end
legend(legendString);
title(['Constrained Capacity for ', MOD_TYPE, ' on ', channelModel, ' channel']);
xlabel('SNR (dB)'); ylabel('Capacity (bits/sym)');
grid on;

function [r, n, N0] = add_awgn_noise(s, SNRdB, L)
% Function to add AWGN to the given signal
% [r, n, N0]= add_awgn_noise(s, SNRdB) adds AWGN noise vector to signal
% 's' to generate a resulting signal vector 'r' of specified SNR in dB.
% It also returns the noise vector 'n' that is added to the signal 's'
% and the spectral density N0 of noise added.
%
% [r, n, N0]= add_awgn_noise(s, SNRdB, L) adds AWGN noise vector to signal
% 's' to generate a resulting signal vector 'r' of specified SNR in dB.
% The parameter 'L' specifies the oversampling ratio used in the system
% (for waveform simulation). It also returns the noise vector 'n' that is 
% added to the signal 's' and the spectral density N0 of noise added.
s_temp = s;
if iscolumn(s), s = s.'; end % To return the result in same dim as 's'
gamma = 10^(SNRdB/10); % SNR to linear scale
if nargin == 2, L = 1; end % If third argument is not given, set it to 1

if isvector(s)
    P = L*sum(abs(s).^2)/length(s); % Actual power in the vector
else % For multi-dimensional signals like MFSK
    P = L*sum(sum(abs(s).^2))/length(s); % If s is a matrix [MxN]
end

N0 = P/gamma; % Find the noise spectral density
if isreal(s)
    n = sqrt(N0/2)*randn(size(s)); % Computed noise
else
    n = sqrt(N0/2)*(randn(size(s)) + 1i*randn(size(s))); % Computed noise
end
r = s + n; % Received signal
if iscolumn(s_temp), r = r.'; end % Return r in original format as s
end

function [s, ref] = modulate(MOD_TYPE, M, d, COHERENCE)
% Wrapper function to call various digital modulation techniques
% MOD_TYPE - 'PSK', 'QAM', 'PAM', 'FSK'
% M - modulation order, For BPSK M=2, QPSK M=4, 256-QAM M=256 etc..,
% d - data symbols to be modulated drawn from the set {1,2,...,M}
% COHERENCE - only applicable if FSK modulation is chosen
%           - 'coherent' for coherent MFSK
%           - 'noncoherent' for non-coherent MFSK
% s - modulated symbols
% ref - ideal constellation points that could be used by an IQ detector
switch lower(MOD_TYPE)
    case 'bpsk'
        [s, ref] = mpsk_modulator(2, d);
    case 'psk'
        [s, ref] = mpsk_modulator(M, d);
    case 'qam'
        [s, ref] = mqam_modulator(M, d);
    case 'pam'
        [s, ref] = mpam_modulator(M, d);
    case 'fsk'
        [s, ref] = mfsk_modulator(M, d, COHERENCE);
    otherwise
        error('Invalid Modulation specified');
end
end

function [s, ref] = mpsk_modulator(M, d)
% Function to MPSK modulate the vector of data symbols - d
% [s, ref]=mpsk_modulator(M, d) modulates the symbols defined by the
% vector d using MPSK modulation, where M specifies the order of
% M-PSK modulation and the vector d contains symbols whose values
% in the range 1:M. The output s is the modulated output and ref
% represents the reference constellation that can be used in demod
ref_i = 1/sqrt(2)*cos(((1:1:M)-1)/M*2*pi);
ref_q = 1/sqrt(2)*sin(((1:1:M)-1)/M*2*pi);
ref = ref_i + 1i*ref_q;
s = ref(d); % M-PSK Mapping
end

function [s, ref] = mqam_modulator(M, d)
% Function to MQAM modulate the vector of data symbols - d
% [s, ref]=mqam_modulator(M, d) modulates the symbols defined by the
% vector d using MQAM modulation, where M specifies the order of
% M-QAM modulation and the vector d contains symbols whose values
% range 1:M. The output s is the modulated output and ref
% represents the reference constellation that can be used in demod
if((M~=1) && ~mod(floor(log2(M)),2)==0)
    error('Only Square MQAM supported. M must be even power of 2');
end
ref = constructQAM(M); % Construct reference constellation
s = ref(d); % Map information symbols to modulated symbols
end

function [s, ref] = mpam_modulator(M, d)
% Function to MPAM modulate the vector of data symbols - d
% [s, ref]=mpam_modulator(M, d) modulates the symbols defined by the
% vector d using MPAM modulation, where M specifies the order of
% M-PAM modulation and the vector d contains symbols whose values
% in the range 1:M. The output s is the modulated output and ref
% represents the reference constellation
m = 1:1:M;
Am = complex(2*m-1-M); % All possible amplitude levels
s = complex(Am(d)); % M-PAM transmission
ref = Am; % Reference constellation
end

function [s, ref] = mfsk_modulator(M, d, COHERENCE)
% Function to MFSK modulate the vector of data symbols - d
% [s, ref]=mfsk_modulator(M, d, COHERENCE) modulates the symbols defined
% by the vector d using MFSK modulation, where M specifies the order
% of M-FSK modulation and the vector d contains symbols whose values
% in the range 1:M.
% The parameter 'COHERENCE' = 'COHERENT' or 'NONCOHERENT' specifies
% the type of MFSK modulation/detection. The output s is the
% modulated output and ref represents the reference constellation
% that can be used during coherent demodulation.
if strcmpi(COHERENCE, 'coherent')
    phi = zeros(1, M); % Phase=0 for coherent detection
    ref = complex(diag(exp(1i*phi))); % Force complex data type
    s = complex(ref(d, :)); % Force complex type, since imag part is 0
else
    phi = 2*pi*rand(1, M); % M random phases in the (0,2pi)
    ref = diag(exp(1i*phi));
    s = ref(d, :);
end
end

function [varargout] = constructQAM(M)
% Function to construct gray codes symbol constellation for M-QAM
% [ref]=constructQAM(M) - returns the ideal signaling points (ref)
% in a symmetric rectangular M-QAM constellation, where M is the
% level of QAM modulation. The returned constellation points are
% arranged such that the index of the points are arranged in a
% Gray-coded manner. When plotted, indices of constellation points
% will differ by 1 bit.
%
% [ref, I, Q]=constructQAM(M) - returns the ideal signaling points
% (ref) along with the IQ components breakup in a symmetric
% rectangular M-QAM constellation, where M is the level of QAM
% modulation. The returned constellation points are arranged such
% that the index of the points are arranged in a Gray-coded manner.
% When plotted, the indices of the constellation points will
% differ by 1 bit.
n = 0:1:M-1; % Sequential address from 0 to M-1 (1xM dimension)

%------Addresses in Kmap - Gray code walk---------------
a = dec2gray(n); % Convert linear addresses to gray code
N = sqrt(M); % Dimension of K-Map - N x N matrix
a = reshape(a, N, N).'; % NxN gray coded matrix
evenRows = 2:2:size(a, 1); % Identify alternate rows
a(evenRows, :) = fliplr(a(evenRows, :)); % Flip rows - KMap representation
nGray = reshape(a.', 1, M); % Reshape to 1xM - Gray code walk on KMap

% Construction of ideal M-QAM constellation from sqrt(M)-PAM
D = sqrt(M); % Dimension of PAM constellation
x = floor(nGray / D);
y = mod(nGray, D);
Ax = 2*(x + 1) - 1 - D; % PAM Amplitudes 2m-1-D - real axis
Ay = 2*(y + 1) - 1 - D; % PAM Amplitudes 2m-1-D - imag axis
varargout{1} = Ax + 1i * Ay; % M-QAM points (I+jQ)
varargout{2} = Ax; % Real part (I)
varargout{3} = Ay; % Imaginary part (Q)
end

function [grayCoded] = dec2gray(decInput)
% Convert decimal to Gray code representation
% Example: x= [0 1 2 3 4 5 6 7] %decimal
% y=dec2gray(x)
% Returns y = [ 0 1 3 2 6 7 5 4] %Gray coded
[rows, cols] = size(decInput);
grayCoded = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        grayCoded(i, j) = bitxor(bitshift(decInput(i, j), -1), decInput(i, j));
    end
end
end