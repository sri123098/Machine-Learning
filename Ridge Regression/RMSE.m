function p = RMSE(W,A,B)
    total=size(B);
    total=total(1,1);
    k=transpose(W)*A - transpose(B);
    p = sqrt(k*transpose(k)/total);
    %Perfectly I'm implementing Root Mean Square Error
end