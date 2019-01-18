function [centroid]=KNN(X,p3)
[n,d]=size(X);
for k=[p3]
p=(randi([1,n],k,1));
pick=1:k;
% centroid=X(p,:);
centroid=X(pick,:);
iter=1:20;
SSS=[];
VV=zeros(n,1);
% for i = iter
for count = iter
    V=zeros(n,1);
    for j=1:n
% Instead of using the for use matrix. and argmax of the index.
% Each example I have to compare with respect to all the k clusters.
        D= centroid - X(j,:);
        [minimum,index]=min(sum((D.^2)'));
        V(j)=index;
    end
% Checking the assignment for the break condition
    if VV(:,end)==V
%         fprintf(" Iteration for breaking %d \n", count);
        break;
    else
        VV=[VV V];  
    end
%We are calculating the percentages only at the end.
%Calculation of SS(k)
% group sum of squares per cluster
SS=0;
for i=1:k
% Only one-dimensional indexing is supported for c(i)
    Cluster_mean=mean(X(V==i,:));
    [row col]=size(X(V==i,:));
%     fprintf(" Row %d & Column %d \n", row, col);
    D=X(V==i,:) - Cluster_mean;
    [row col]=size(D);
%     fprintf(" Row %d & Column %d \n", row, col);
    SS=SS+sum(sum((D.^2)')); 
end
SSS=[SSS SS];
%Calculation of new centroid
    for i=1:k
%         disp(i);
        dim=size(X(V==i,:));
%         fprintf("Cluster %d & its elements %d Iteration %d \n", i,dim(1),count);
        centroid(i,:)=mean(X(V==i,:));
    end
end

centroid=centroid'
end



    