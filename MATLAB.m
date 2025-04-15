% linprog
% https://www.mathworks.com/help/optim/ug/linprog.html
% https://www.mathworks.com/help/optim/ug/choosing-the-algorithm.html

% Load LP data
folderName = '\MATLAB';  
A = load(fullfile(folderName, 'A_matrix.txt'));
b = load(fullfile(folderName, 'b_vector.txt'));
f = load(fullfile(folderName, 'c_vector.txt'));
lb = zeros(size(f));

%options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');

    
% Solve and time
tStart = tic;
[x, fval, exitflag, output] = linprog(f, [], [], A, b, lb, [], options);
tElapsed = toc(tStart);
    
% solves min f'*x such that A*x = b and x >= 0
