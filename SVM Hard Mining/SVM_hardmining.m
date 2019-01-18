function [w,b] = SVM_hardmining(C,trD,valD,trLb,valLb)
    [d,n] = size(trD);
%     disp("first")
    [nl,pl]=size(valLb);
% nl is used at the end
%     disp("second")
    A = [];
    b = [];
    Aeq = trLb'; 
    beq = 0; % scalar 0
    lb=zeros(n,1);
    ub = C*ones(n,1);
    f = ones(n,1)*-1;
    H = zeros(n,n);
%     disp("third")
    H=(trLb*trLb').*(trD'*trD);
    H=double(H);
    Aeq=double(Aeq);
%     disp("forth")
    [x,obj] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    [max_alpha, index] = max(x);
    w = trD * (trLb.*x);
    b= trLb(index) - w'*trD(:,index);
    Test_result = valD' * w + b;
    Test_result(Test_result<0)=-1;
    Test_result(Test_result>0)=1;
    fprintf("C used in this experiment %d \n",C);
    fprintf("Accuracy is %d \n",sum(Test_result==valLb)/nl);
    fprintf("Dual Objective function value is %d \n",obj);
    alpha_thr = 0.00000001;
    fprintf("Number of support Vectors are %d \n",sum(x > alpha_thr ));
end