%% 2.3
%% No momentum
% net(0, 10, 70, 0.002, 0, false, 4)
% Output: 2.304283
% net(0, 10, 70, 0.01, 0, false, 4)
% Output: 2.302117
% net(0, 10, 70, 0.05, 0, false, 4)
% Output: 2.292967
% net(0, 10, 70, 0.2, 0, false, 4)
% Output: 2.228969
% net(0, 10, 70, 1.0, 0, false, 4)
% Output: 1.598844
% net(0, 10, 70, 5.0, 0, false, 4)
% Output: 2.301322
%% Momentum
% net(0, 10, 70, 0.002, 0.9, false, 4)
% Output: 2.300135
% net(0, 10, 70, 0.01, 0.9, false, 4)
% Output: 2.284022
% net(0, 10, 70, 0.05, 0.9, false, 4)
% Output: 2.008606
% net(0, 10, 70, 0.2, 0.9, false, 4)
% Output: 1.083429
% net(0, 10, 70, 1.0, 0.9, false, 4)
% Output: 2.018723
% net(0, 10, 70, 5.0, 0.9, false, 4)
% Output: 2.302585
%% 2.4
%% No early stopping
% net(0, 200, 1000, 0.35, 0.9, false, 100)
%% Early stopping
% net(0, 200, 1000, 0.35, 0.9, true, 100)
%% WD
% net(10, 200, 1000, 0.35, 0.9, false, 100)
% Output: 22.612774
% net(1, 200, 1000, 0.35, 0.9, false, 100)
% Output: 2.302585
% net(0.0001, 200, 1000, 0.35, 0.9, false, 100)
% Output: 0.348294
% net(0.001, 200, 1000, 0.35, 0.9, false, 100)
% Output: 0.287910
% net(5, 200, 1000, 0.35, 0.9, false, 100)
% Output: 2.302585

%% Hidden layer size without early stopping
% net(0, 10, 1000, 0.35, 0.9, false, 100)
% Output: 0.421705
% net(0, 30, 1000, 0.35, 0.9, false, 100)
% Output: 0.317077
% net(0, 100, 1000, 0.35, 0.9, false, 100)
% Output: 0.368593
% net(0, 130, 1000, 0.35, 0.9, false, 100)
% Output: 0.397597
% net(0, 200, 1000, 0.35, 0.9, false, 100)
% Output: 0.430185
%% Hidden layer size with early stopping
% net(0, 18, 1000, 0.35, 0.9, true, 100)
% Output: 0.306083
% net(0, 37, 1000, 0.35, 0.9, true, 100)
% Output: 0.265165
% net(0, 83, 1000, 0.35, 0.9, true, 100)
% Output: 0.311244
% net(0, 113, 1000, 0.35, 0.9, true, 100)
% Output: 0.313749
% net(0, 236, 1000, 0.35, 0.9, true, 100)
% Output: 0.343841
