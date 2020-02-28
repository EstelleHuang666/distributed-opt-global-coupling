% Number of iterations
nit = 1.3 * 1e5;

% Number of agents
n = 50;

% Seed of the random generator. This way we generate the same "random"
% vectors in each simulation
rng(234)

% Problem data
%c = rand(n,1); % This line must be after the random generator rng(234)
c = (1:n)';
%d = ones(n,1);
d = rand(n,1);

% TEST: take care with this parameter elsewhere
%e = ones(n,1);
e = rand(n,1);

%% 
% Description of the optimization problem

% minimize \sum_i=1^100 c_i x_i 

% subject to \sum_i=1^100 - d_i log( 1 + e_i x_i ) \le -n/10 , 
% x_i\in[0, 1] 


% Recall that n is the number of agents


% The Lagrangian is 
% L(x,mu)=\sum_i=1^100 ( c_i x_i - mu d_i log( 1 + e_i x_i ) )

% Centralized solution using "optimtool" with solver "fmincon" and
% algorithm "interior point"

x0 = zeros(n,1);

lb = zeros(1,n);
ub = ones(1,n);

MaxIter_Data = 1000;
MaxFunEvals_Data = 10000;

%% Start with the default options
options = optimoptions('fmincon');
%% Modify options setting
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'MaxFunEvals', MaxFunEvals_Data);
options = optimoptions(options,'MaxIter', MaxIter_Data);
[x,fval,exitflag,output,lambda,grad,hessian] = ...
fmincon(@(x)objective_mytest(x,n,c),x0,[],[],[],[],lb,ub,@(x)constraint_mytest(x,n,d,e),options);

% The result for the problem in question is 

costfunction = fval;
output_matlab_solver = output;

% Example for n=5 agents with c=1:n, d=ones(n,1), and d=ones(n,1) 
% costfunction = 0.6487312706403711

% (TEST!)
% Evaluation of objective function at the feasible point {x_i=1}
%qbar = sum(c) - sum(c(qq)'*log(2)); 

% Upper bound for the optimal dual set
%D = -2*qbar/(n/10);
D=100;

%%
% Construction of the adjacency matrix. This block uses the function 
% small(), which in turn uses the function short()

% Seed of the random generator (to have predictable outcomes)
rng(234)
% Unweighted adjacency matrix
A = smallw(n, 1, 0.1);
% Build weighted adjacency matrix W with Metropolis weights
W = zeros(n,n);
for ii = 1:n
    for jj = ii + 1:n
        if A(ii,jj) == 1
            W(ii,jj) = 1/(1 + max(sum(A(ii,:)),sum(A(:,jj))));
            W(jj,ii) = W(ii,jj);
        end
    end
end
for ii = 1:n
    W(ii,ii) = 1 - sum(W(ii,:));
end
P = W - ones(n,n)/n;
nu = norm(W - ones(n,n)/n);

%(TEST) The power makes the matrix more populated (like considering phi hop
%communication).
phi=26; %Choose phi value

% Power phi of the adjacency matrix to compute a new adjacency matrix more
% populated
Wphi = W^phi;
