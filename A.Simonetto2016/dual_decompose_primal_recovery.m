%%
% Distributed solution with CoBa-DD. 
% Primal Recovery from Consensus-Based Dual Decomposition for Distributed
% Convex Optimization, by
% Andrea Simonetto and Hadi Jamali-Rad


% Primal and dual variables.
xk = zeros(n, 1); 

% (Recall that each agent maintains a copy of the multiplier)
muk = zeros(n,1);

% Objective values at each iteration
fk = zeros(nit,1);

% Constraint values at each iteration
constraint_CoBA_DD = zeros(nit, 1);

% (Sub-) gradient stepsize
eta_d = .01*n;


for t=1:nit
    
    % xktilde is the auxiliary primal variable before computing ergodic
    % sums
    xktilde = zeros(n,1);
    
    
    % Update of auxiliary primal variables.
    % Each coordinate is the minimizer of the Lagrangian
    % for a given muk.
    xktilde = ( muk .* d .* e  - c) ./ (c .* e); 
    
    % projection onto the set [0, 1]
    xktilde = max(min(xktilde ,1 ), 0);
    
    % Ergodic sum to recover the primal variable from the auxiliary primal
    % variables
    xk = xk*(t-1)/t + xktilde/t;
    
    % Calculate the constraint value
    constraint_CoBA_DD(t) = sum(-d .* log(1 + e .* xktilde)) + n/10;
    
    % Update of the multipliers
    muk = Wphi*(muk + eta_d*(-d .* log(1 + e .* xktilde) + n/10/n + 1e-8));
    
    % Projection of multipliers onto the set [0, D]
    muk = min(max(0, muk), D); 
    
    % Update the cost
    fk(t) = c'*xk;
    
end
