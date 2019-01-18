function [check] = LOOCV_final(W,C,X,Y)
%     X and Y are column vectors. 
    total=size(X);
    total=total(1,2);
%     disp(total)
%     W_I=[];
%     E_I=[];
    D_I=[];
    for i=linspace(1,total,total)
        x_i=X(:,i);
        y_i=Y(i,:);
        scalar1= transpose(x_i)*W - y_i;
        scalar2= 1-transpose(x_i)*C*x_i;
%         w_i = W + C*x_i*(scalar1/scalar2);
%         error_i=transpose(w_i)*x_i - y_i;
        direct = transpose(W)*x_i - y_i;
        direct = direct/scalar2 ;
        D_I=[D_I,direct];
%         W_I=[W_I,w_i];
%         E_I=[E_I,error_i];  
    end
%     disp(D_I(1))
%     disp(D_I(2))
%     disp(D_I(3))
%     disp(D_I(3997))
%     disp(D_I(3998))
%     disp(D_I(3999))
%     OCV=sqrt((E_I*transpose(E_I))/total);
    check=sqrt((D_I*transpose(D_I))/total);
end