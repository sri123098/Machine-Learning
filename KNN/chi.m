function [b]=chi(mat1,mat2,gamma)
% mat1 is of dimension d*n
% mat2 is of dimension d*m
% Out put is of dimension n*m 
[d n]=size(mat1);
[d m]=size(mat2);
epsilon=0.000000001;
% b=zeros(size(mat1,2),size(mat2,2));
% b=zeros(n,m);
b=[];
for i=1:n
    D=mat1(:,i)-mat2;
    D=D.^2;
    E=mat1(:,i)+mat2+epsilon;
    f=exp(-(1/gamma)*sum(D./E));
    b=[b;f];
end
end

