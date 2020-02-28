%%
% Simulation of Consensus-Based Primal Dual Perturbed algorithm
% Distributed Constrained Optimization by Consensus-Based Primal-Dual 
% Perturbation Method 
% by Tsung-Hui Chang and Angelia Nedic and Anna Scaglione


% (TEST)
% Gradient step sizes for primal and dual variables
eta_p = zeros(1, nit+1);
eta_p = 0.1./(10+(1: nit+1));
%eta_p= 0.1*eta_d*ones(1,nit+1);

%(TEST)
% Gradient step sizes for perturbation points
rho_1 = 0.0001;
rho_2 = 0.0001;  

% Initialize the cummulative sum of learning rates
sum_eta_p    = zeros(1,nit+1);
sum_eta_p(1) = eta_p(1);

% Perturbatin points (p stands for "perturbed") for primal and dual
% variables
alpha_p = zeros(n,1);
beta_p  = zeros(n,1);

% Primal auxiliary variable (before computing the running weighted average)
xtilde_p = rands(n, 1);

% xtilde_p_plus is the update of xtilde_p (defined for convenience)
% xtilde_p_plus

% Primal and dual variables
% x_p      = zeros(n, 1);
lambda_p = rands(n, 1); % Plays the role of mu in previous algorithms

% Extra auxiliary variable that appears in the computation of beta_p
% z_p = zeros(n, 1); % (This is not a multiplier estimate)

% This variables need to be initialized as the constraint evaluated at the
% primal variables
z_p = -d .* log(1+ e .* xtilde_p) + n/10/n;

% Constraint values at each iteration
constraint_PDP = zeros(nit, 1);

% Objective values at each iteration
fk_p = zeros(nit, 1);


for t=1:nit
    
    % 1) AVERAGE CONSENSUS
    
    ztilde_p      =  Wphi * z_p;
    
    lambdatilde_p =  Wphi * lambda_p;

% (TEST) What happens if we use the Laplacian with the same stepsize?
%     ztilde_p      = ( eye(n) - sig*Lap ) * z_p;
%     
%     lambdatilde_p = ( eye(n) - sig*Lap ) * lambda_p;
    
    
    % 2) PERTURBATION POINT COMPUTATION
    % (They use gradient descent when functions are smooth)
    
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable evaluated at primal and dual variables
    bFby = sum(1/n^2 .* 1./ Wphi, 2);
    g_alpha_p = c.*bFby - (lambdatilde_p .* d .* e) ./(1 + e.* xtilde_p);
    
    
    % Perturbation point for primal variable
    alpha_p = xtilde_p - rho_1 * g_alpha_p;
    
    %Projection of alpha_p onto [0, 1]
    alpha_p = min(max(0, alpha_p), 1);
   
    % Perturbation point for dual variable
    beta_p = lambdatilde_p + rho_2 * n * ztilde_p;
    
    % Projections of beta_p onto [0, D]
    beta_p = min(max(0, beta_p), D);
    
    % 3) PRIMAL-DUAL PERTURBED SUBGRADIENT UPDATE
    
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable evaluated at primal variable & perturbation point for the
    % dual variable
    g_xtilde_p    = c.*bFby - (beta_p .* d .* e) ./(1 + e.* xtilde_p);
    
    % Update of primal (auxiliary) variable using a (minus) subgradient
    % step
    xtilde_p_plus = xtilde_p - eta_p(t) * g_xtilde_p  ;
    
    % Projection step onto [0, 1]
    xtilde_p_plus = min(max(0, xtilde_p_plus), 1);
    
    % Update of the multipliers
    % Subgradient (column vector) of Lagrangian with respect to dual
    % variable (i.e., the constraint function) evaluated at the 
    % perturbation point for the primal variable.
    % Note that lambdatilde_p already contains the average
    lambda_p = lambdatilde_p...
        + eta_p(t) * (-d .* log(1+ e .* alpha_p) + n/10/n);
    
    
    
    % projection of multipliers onto the set [0, D]
    lambda_p = min(max(0, lambda_p), D); 
    
    % 4) AUXILIARY VARIABLES THAT APPEAR IN PERTURBATION POINT COMPUTATIONS
    
    % difference of constraint evaluated at xtilde and xtilde in previous
    % iteration
    z_p = ztilde_p + (-d .* log(1+ e .* xtilde_p_plus) + n/10/n) ...
        -(-d .* log(1+ e .* xtilde_p) + n/10/n);

    
    % 6) Preparing for next iteration
    
    % Update old estimate
    xtilde_p = xtilde_p_plus;
    
    % Update the cost
    fk_p(t) = c'*xtilde_p;

    % Calculate the constraint value
    constraint_PDP(t) = sum(-d .* log(1 + e .* xtilde_p)) + n/10;

end

