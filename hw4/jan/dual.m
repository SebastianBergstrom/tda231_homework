clear all
t = [1,1,1,-1,-1,-1];
x = [2,2;4,4;4,0;0,0;2,0;0,2];
N = length(t);
alpha = fmincon(@(alpha) -f(alpha, t, x), zeros(N,1), zeros(1,N), 0, t, 0, zeros(N,1))
