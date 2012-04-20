%
% Implements the worked example given at the start of
%   Longstaff, F.A. & Schwartz, E.S. Valuing American options by simulation:
%   A simple least-squares approach. Review of Financial Studies 14, 113–147 (2001).
%

r = 0.06;
T = 3;
K = 3;
N = 8;
strike = 1.10;

paths = [
    1 1.09 1.08 1.34;
    1 1.16 1.26 1.54;
    1 1.22 1.07 1.03
    1 0.93 0.97 0.92;
    1 1.11 1.56 1.52;
    1 0.76 0.77 0.90;
    1 0.92 0.84 1.01;
    1 0.88 1.22 1.34;
]';

t = linspace(0, T, K+1)';
times = repmat(t, [1 N]);
cashflows = zeros(K, N);
cashflows(K, :) = max(strike - paths(K+1,:), 0);

basis = @(x)[ones(size(x')) x' x'.^2];

for ii = K-1:-1:1
    exercisevalue = max(strike - paths(ii+1,:), 0);
    isITM = (exercisevalue > 0);
    futurestoppingtimes = max(times(ii+2:end, isITM), [], 1);
    futurestoppingcashflows = max(cashflows(ii+1:end, isITM), [], 1);
    discountedfuturecashflows = exp(-r*(futurestoppingtimes - t(ii+1))) .* futurestoppingcashflows

    % Perform regression.
    X = paths(ii+1, isITM);
    b = regress(discountedfuturecashflows', basis(X));

    continuationvalue = (basis(paths(ii+1,:))*b)';
    shouldExercise = (exercisevalue > continuationvalue) & isITM;

    % Setup cash flows for paths exercised at this time point and
    % clear any cash flows and stopping times in the future.
    cashflows(ii, shouldExercise) = exercisevalue(shouldExercise);
    cashflows(ii+1:end, shouldExercise) = 0;
    times(ii+2:end, shouldExercise) = 0;

    cashflows(ii, ~shouldExercise) = 0;
    times(ii+1, ~shouldExercise) = 0;
end

stoppingmask = times(2:end, :) > 0;
stoppingtimes = max(times, [], 1);
price = mean(exp(-r*stoppingtimes) .* cashflows(stoppingmask)')
