% 1. Select k starting centers that are points from your data set. You should be able to select these centers
% randomly or have them given as a parameter.
% 2. Assign each data point to the cluster associated with the nearest of the k center points.
% 3. Re-calculate the centers as the mean vector of each cluster from (2).
% 4. Repeat steps (2) and (3) until convergence or iteration limit.
X=load('/Users/sriramreddy/Downloads/ML/hw5data/digit/digit.txt');
Y=load('/Users/sriramreddy/Downloads/ML/hw5data/digit/labels.txt');
[n,d]=size(X);
pas1=[];
pas2=[];
pas3=[];
SSa3=[];


for cap=1:10
    
ps1=[];
ps2=[];
ps3=[];
SS3=[];
    
for k=1:10
p=(randi([1,n],k,1));
pick=1:k;
centroid=X(p,:);
% centroid=X(pick,:);
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

%Percentage of the pairs
p1=0;
p2=0;
total_p1=0;
total_p2=0;
for i=1:n
    for j=i+1:n
      if  Y(i)==Y(j)
         total_p1 = total_p1 + 1;
         if V(i)==V(j)
          p1=p1+1;
         end
      elseif Y(i)~=Y(j) 
          total_p2 = total_p2 + 1;
          if V(i)~=V(j)
            p2=p2+1;
          end
      end
    end
end
format;
p1=(p1/total_p1)*100;
p2=(p2/total_p2)*100;
p3=(p1+p2)/2;


ps1=[ps1 p1];
ps2=[ps2 p2];
ps3=[ps3 p3];
SS3=[SS3 SSS(end)];
fprintf("Clusters %d, Iteration break %d, SS(k)=%d, p1=%.2f, p2=%.2f & p3 %.2f \n", k,count, SSS(end),p1,p2,p3);

% fprintf("K means %d Iteration break %d SS(k) %d p1 %d p2 %d & p3 %d \n", k, count, SSS(end),p1,p2,p3);
end
pas1=[pas1;ps1];
pas2=[pas2;ps2];
pas3=[pas3;ps3];
SSa3=[SSa3;SS3];


end

ps1=mean(pas1);
ps2=mean(pas2);
ps3=mean(pas3);
SS3=mean(SSa3);
K=1:10;
% figure(1)
% plot(K,ps1,'r*-')
% xlabel('K means 1 to 10')
% ylabel('p1')
% title('p1')
% hold on;
% 
% grid
% plot(K,ps2,'g*-')
% xlabel('K means 1 to 10')
% ylabel('p2')
% title('p2')
% hold on;
% grid
% xlabel('K means 1 to 10')
% plot(K,ps3,'b*-')
% ylabel('p3')
% title('p3')
% grid

figure(1)
plot(K,ps1,'r*-')
xlabel('K means 1 to 10')
hold all;
grid
plot(K,ps2,'g*-')
xlabel('K means 1 to 10')
hold all;
grid
xlabel('K means 1 to 10')
plot(K,ps3,'b*-')
ylabel('p1 p2 p3 %')
legend('p1','p2','p3')
title('p1 p2 p3')
grid


figure(2)

plot(K,SS3,'b*-')
ylabel('Group sum of squares')
xlabel('K means 1 to 10')
title('Group sum of squares')
grid
          
% How to find the percentage of the pairs?
% I need to get the distance minimum and then I need to get the index
% Once I get the index, I have to update the Valid.
% If I'm proceeding in this fashion, calculation of the centroid.
% I thought that it is tough 
% Matrix indexing property (Y==1) and then take mean.
% I need to implement the break condition as well.
% If there is no change in the labels or if it meets maximum number of iterations.







