function [w,b,obj] = SVM_final10(trD,trLb,C)
    [d,n] = size(trD);
    A = [];
    b = [];
    Aeq = trLb'; 
    beq = 0; % scalar 0
    lb=zeros(n,1);
    ub = C*ones(n,1);
    f = ones(n,1)*-1;
    H = zeros(n,n);
    H=(trLb*trLb').*(trD'*trD);
    H=double(H);
    Aeq=double(Aeq);
    [x,obj] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    [max_alpha, index] = max(x);
    w = trD * (trLb.*x);
    b2= trLb(index) - w'*trD(:,index);
    for i=1:n
        if ((C - x(i)) > 0.0001 & (x(i) - 0.0) > 0.0001)
            b= trLb(i) - w'*trD(:,i);
            break;
        end
    end
    fprintf("b values max and ppt %d %d \n", b2,b);
    fprintf("Dual Objective function value is %d %d \n",obj);
    alpha_thr = 0.00000001;
    fprintf("Number of support Vectors are %d \n",sum(x > alpha_thr ));
end
