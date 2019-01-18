function [w,b,acc] = SVM_final5(trD,valD,trLb,valLb,filenames,C,count,gamma)
    data=load(filenames{1});
    [d,n] = size(trD);
    [nl,pl]=size(valLb);
    trLb(trLb~=count)=-1;
    valLb(valLb~=count)=-1;
    trLb(trLb==count)=1;
    valLb(valLb==count)=1;
%     trLb=[trLb;valLb];
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
% The below code should be commented
%     gamma=0.04;
    for i=1:n
    for j=1:n
         rbf=exp(-sum((trD(:,i)-trD(:,j)).^2)*gamma);
         H(i,j) = trLb(i)*trLb(j)*rbf; 
    end
    end
%     k(x1,x2) = exp(-sum((trD(;,i)-trD(;,j)).^2)./sigma^2);
%     H=(trLb*trLb').*(trD'*trD)^2;
    
    H=double(H);
    Aeq=double(Aeq);
%     disp("forth")
    [x,obj] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    [max_alpha, index] = max(x);
    fprintf("max_alpha value and C value are %d %d \n", max_alpha, C);
    w = trD * (trLb.*x);
    b= trLb(index) - w'*trD(:,index);
    Test_result = valD' * w + b;
    Test_result(Test_result<0)=-1;
    Test_result(Test_result>0)=1;
    fprintf("C used in this experiment %d \n",C);
%      Accuracy is nothing but the correct predictions by total number of
%      predictions
    acc=sum(Test_result==valLb)/nl;
    fprintf("Accuracy is %d %d \n", count,acc);
    fprintf("Dual Objective function value is %d %d \n",count,obj);
    alpha_thr = 0.00000001;
    
    fprintf("Number of support Vectors are %d %d \n",count,sum(x > alpha_thr ));
end