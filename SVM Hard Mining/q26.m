filenames={"/Users/sriramreddy/Downloads/ML/hw4data/all/q2_2_data.mat"};
C=10;
ls=[];
ps=[];
data=load("/Users/sriramreddy/Downloads/ML/hw4data/all/q2_2_data.mat");
trD=data.trD;
% trD = normalize(trD);
valD=data.valD;
% trD = [trD';valD'];
% trD=trD';
% valD = normalize(valD);
% trD = [trD';valD'];
% trD=trD';
% l=sum(trD);
% trD=[trD; l];
% p=l.*l;
% trD=[trD;p];
% trD=normalize(trD);
%     for C=[0.01,0.1,1,10,100,1000]
%         for g=[0.001,0.01,0.1,1,10,100,1000]

% 2,9,10,7  I need to improve accuracy for them


Acc=[];
for i=2
    for C=[0.01,0.001]
            data=load("/Users/sriramreddy/Downloads/ML/hw4data/all/q2_2_data.mat");
            trLb=data.trLb;
            valLb=data.valLb;
            [w,b,acc] = SVM_final7(trD,valD,trLb,valLb,C,i,g);
            ls=[ls,w];
            ps=[ps,b];
            Acc=[Acc,acc]; 
    end
end
data=load(filenames{1});
tstD=data.tstD;
% lp=sum(tstD);
% tstD=[tstD; lp];
% t=lp.*lp;
% tstD=[tstD;t];
% tstD=normalize(tstD);
pred=ls'*tstD + ps';
[~,ii]=max(pred);
m=ii;
N=1:3190;
m=[N;m];
csvwrite('myFile2.txt',m');


% data=load(filenames{1});
% tstD=data.tstD;
% % lp=sum(tstD);
% % tstD=[tstD; lp];
% % t=lp.*lp;
% % tstD=[tstD;t];
% tstD=normalize(tstD);
% 
% 
% pred=ls'*tstD + ps';
% [~,ii]=max(pred);
% m=ii;
% N=1:3190;
% m=[N;m];
% csvwrite('myFile2.txt',m');



% ABove code should be uncommented.
% tstD is 4096*(3190---examples)
% w is of 4096*1
% what cna i do ?
% 10*4096 * 4096*3190 ---- 10*3190 + 10(bias value)
%w=[1,2,3;3,8,9;3,6,7]
% w =
%      1     2     3
%      3     8     9
%      3     6     7
% w+[1;2;3]
% ans =
%      2     3     4
%      5    10    11
%      6     9    10 
% [argvalue, argmax] = max(x);
% Accuracy which I'm getting is 96% and 90% 
% max_alpha value and C value are 2.495595e-03 10 
% C used in this experiment 10 
% Accuracy is 1 9.504717e-01 
% Dual Objective function value is 1 -3.399123e-02 
% Number of support Vectors are 1 176 
% max_alpha value and C value are 1.002490e-02 10 
% C used in this experiment 10 
% Accuracy is 2 9.000000e-01 
% Dual Objective function value is 2 -9.333409e-02 
% Number of support Vectors are 2 276 
% max_alpha value and C value are 3.009098e-03 10 
% C used in this experiment 10 
% Accuracy is 3 9.603774e-01 
% Dual Objective function value is 3 -4.592127e-02 
% Number of support Vectors are 3 307

% *******Experiment 2*****
% 
% for i=1
%     for C=[0.1, 10, 50, 100,1000]
%         data=load("/Users/sriramreddy/Downloads/ML/hw4data/all/q2_2_data.mat");
%         trLb=data.trLb;
%         valLb=data.valLb;
%         [w,b] = SVM_final(trLb,valLb,filenames,C,i);
%         ls=[ls,w];
%         ps=[ps,b];
%     end
% end
% There is no impact of C on this. 
% here are the results after changing the C value. 
% max_alpha value and C value are 2.495517e-03 1.000000e-01 
% C used in this experiment 1.000000e-01 
% Accuracy is 1 9.504717e-01 
% Dual Objective function value is 1 -3.399042e-02 
% Number of support Vectors are 1 189 
% 
% max_alpha value and C value are 2.495595e-03 10 
% C used in this experiment 10 
% Accuracy is 1 9.504717e-01 
% Dual Objective function value is 1 -3.399123e-02 
% Number of support Vectors are 1 176 
% 
% max_alpha value and C value are 2.495240e-03 50 
% C used in this experiment 50 
% Accuracy is 1 9.504717e-01 
% Dual Objective function value is 1 -3.398872e-02 
% Number of support Vectors are 1 220 
% 
% The problem is non-convex.
% max_alpha value and C value are 5.567362e+02 100 
% C used in this experiment 100 
% Accuracy is 1 1.816038e-01 
% Dual Objective function value is 1 7.682288e+14 
% Number of support Vectors are 1 7930 
% 
% The problem is non-convex.
% 
% max_alpha value and C value are 8.474631e+02 1000 
% C used in this experiment 1000 
% Accuracy is 1 5.165094e-01 
% Dual Objective function value is 1 2.078829e+15 
% Number of support Vectors are 1 7930 

% ***Experiment 3***
% I'm using the normalizing of the columns as features. 

% I'm seeing the improvement for 1.
% max_alpha value and C value are 10 10 
% C used in this experiment 10 
% Accuracy is 1 9.636792e-01 
% Dual Objective function value is 1 -2.807372e+02 
% Number of support Vectors are 1 176 
% Let's see if I can see any improvement for the other categories.



% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 10 10 
% C used in this experiment 10 
% Accuracy is 1 9.636792e-01 
% Dual Objective function value is 1 -2.807372e+02 
% Number of support Vectors are 1 176 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 10 10 
% C used in this experiment 10 
% Accuracy is 2 8.985849e-01 
% Dual Objective function value is 2 -6.324077e+02 
% Number of support Vectors are 2 299 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 1.000000e+01 10 
% C used in this experiment 10 
% Accuracy is 3 9.594340e-01 
% Dual Objective function value is 3 -2.975146e+02 
% Number of support Vectors are 3 178

% ****Experiment 4****
% normalize---is improving the results.
% max_alpha value and C value are 2.702408e-03 10 
% C used in this experiment 10 
% Accuracy is 1 9.646226e-01 
% Dual Objective function value is 1 -6.001610e-02 
% Number of support Vectors are 1 194 
% 
% max_alpha value and C value are 6.170880e-03 10 
% C used in this experiment 10 
% Accuracy is 2 9.018868e-01 
% Dual Objective function value is 2 -1.410516e-01 
% Number of support Vectors are 2 301 
% 
% max_alpha value and C value are 3.468240e-03 10 
% C used in this experiment 10 
% Accuracy is 3 9.650943e-01 
% Dual Objective function value is 3 -6.423105e-02 
% Number of support Vectors are 3 186 

% ***Experiment 5 Adding the extra features***
% max_alpha value and C value are 2.022840e+00 10 
% C used in this experiment 10 
% Accuracy is 1 9.481132e-01 
% Dual Objective function value is 1 -4.157891e+01 
% Number of support Vectors are 1 196 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 4.160947e+00 10 
% C used in this experiment 10 
% Accuracy is 2 8.985849e-01 
% Dual Objective function value is 2 -7.625521e+01 
% Number of support Vectors are 2 285 

% max_alpha value and C value are 2.382339e+00 10 
% C used in this experiment 10 
% Accuracy is 3 9.636792e-01 
% Dual Objective function value is 3 -4.034813e+01 
% Number of support Vectors are 3 198 


% Normalization and Feature addition is same as that of normalization
% alone.

%Stand alone feature addition***
% Testing for the linear kernel
% Experiment 10

% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 1.000000e-03 1.000000e-03 
% C used in this experiment 1.000000e-03 
% Accuracy is 1 8.679245e-01 
% Dual Objective function value is 1 -1.878933e+00 
% Number of support Vectors are 1 7930 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 1.000000e-02 1.000000e-02 
% C used in this experiment 1.000000e-02 
% Accuracy is 1 8.929245e-01 
% Dual Objective function value is 1 -1.869329e+01 
% Number of support Vectors are 1 7930 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 1.000000e-01 1.000000e-01 
% C used in this experiment 1.000000e-01 
% Accuracy is 1 9.009434e-01 
% Dual Objective function value is 1 -1.773290e+02 
% Number of support Vectors are 1 7836 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 1 1 
% C used in this experiment 1 
% Accuracy is 1 8.797170e-01 
% Dual Objective function value is 1 -1.013989e+03 
% Number of support Vectors are 1 7721 
% 
% Minimum found that satisfies the constraints.
% 
% Optimization completed because the objective function is non-decreasing in 
% feasible directions, to within the default value of the optimality tolerance,
% and constraints are satisfied to within the default value of the constraint tolerance.
% 
% <stopping criteria details>
% 
% max_alpha value and C value are 1.767635e+00 100 
% C used in this experiment 100 
% Accuracy is 1 9.018868e-01 
% Dual Objective function value is 1 -1.126795e+03 
% Number of support Vectors are 1 7720 











