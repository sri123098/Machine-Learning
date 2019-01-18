% The following code should be for 2.1
data = load("/Users/sriramreddy/Downloads/ML/hw4data/q2_1_data.mat");
% data_2  = load("/Users/sriramreddy/Downloads/ML/hw4data/all/q2_2_data.mat");
trD = data.trD;
trLb = data.trLb;
[d,n] = size(trD);
valD = data.valD;
valLb = data.valLb;
% https://www.mathworks.com/help/matlab/ref/transpose.html
C= 10;
% C=10;
% Since there are no inequality constraints, you can use A and b as empty
% matrix
A = [];
b = [];
Aeq = trLb'; 

% Cx = d Here in this case, summation of y_j,alpha_j =0  alpha_j is x in my
% case
beq = 0; % scalar 0
lb=zeros(n,1);
ub = C*ones(n,1);

% I'm trying to minimize -f(x) ---- same as the maximizing the dual
% objective function. 
% That's why f is taken as negative of alpha_j
f = ones(n,1)*-1;
% The above one is bit simple. Let's tackle the case of H.
% Since alpha_i is ranging from 1 to n 
% x.T * H * x will be satisfied if H is of n*n
% H = zeros(n,n);
% Whole thing goes around the design of H matrix from the product of
% alpha_i*alpha_j*label_i*label_j*Kernel(x_i*x_j)
% For now, I'm taking the kernel as the dot product of x_i and x_j
% for i=1:n
%     for j=1:n
%         H(i,j) = trLb(i)*trLb(j)*dot(trD(:,i),trD(:,j)); 
%     end
% end
H=(trLb*trLb').*(trD'*trD);

[x,obj] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
% x here is my Dual output
% Till this point, I have got only alpha.
% Not sure whether it will be right or wrong.

[max_alpha, index] = max(x);

for i=1:n
    if ((C - x(i)) > 0.0001 & (x(i) - 0.0) > 0.0001)
        b= trLb(i) - w'*trD(:,i);
        break;
    end
end
% Primal solution can be derived from Dual solution using
% Summation over examples alpha_i*y_i*x_i
% w=zeros(d,1);
% for i=1:n
%     w=w+x(i)*trLb(i)*trD(:,i);
% end
% Even the above code is giving the same results.

w = trD * (trLb.*x);
% b = y^k -w · x^k for any k where 0 < ↵k < C.
b2= trLb(index) - w'*trD(:,index);
% I'm getting the value of b
Test_result = valD' * w + b;

% % % Check the following part once
Test_result(Test_result<0)=-1;
% Test_result(Test_result>=0)=1; for C=0.1
Test_result(Test_result>0)=1;
% report the accuracy, the objective value of SVM, the number of support vectors, and the confusion
% matrix.
fprintf("b value through index and ppt %d %d \n",b2,b);
fprintf("C used in this experiment %d \n",C);
[p,t]=size(valLb);
fprintf("Accuracy is %d \n", sum(Test_result==valLb)/p);
fprintf("Dual Objective function value is %d \n",-obj);
%sum(x)- 0.5*(x'*H*x)
% used earlier 0.0000001
alpha_thr = 0.00000001;
fprintf("Number of support Vectors are %d \n",sum(x > alpha_thr ));
cmat = confusionmat(valLb,Test_result);
% https://www.mathworks.com/help/stats/confusionmat.html
fprintf("Confusion Matrix is \n");
disp(cmat);
% fmt = [repmat('%4d ', 1, size(cmat,2)-1), '%4d\n'];
% fprintf(fmt, cmat.');
