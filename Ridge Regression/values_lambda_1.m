% Z and P contains the matrix after removing the first index;
%Zack Pot;
%You might need to remove the first column ;
%For retweaking of the code and reproducability;
%you have to implement LOOCV based on the derivations in the previous part. ;
% Calculate LOOCV error for each λ and plot it alongside training and validation;
% error to identify the best λ.;
% The way they are representing is different ;
% K=3000 features;
% N=4999 examples;
% So, I have to change the format;
% Leave one out cross validation error
% Z and P contains the matrix after removing the first index;
%Zack Pot;
%You might need to remove the first column ;
%For retweaking of the code and reproducability;
%you have to implement LOOCV based on the derivations in the previous part. ;
% Calculate LOOCV error for each λ and plot it alongside training and validation;
% error to identify the best λ.;
% The way they are representing is different ;
% K=3000 features;
% N=4999 examples;
% So, I have to change the format;
% Leave one out cross validation error
X = csvread('/Users/sriramreddy/Downloads/ML/2/all',0,0);
Y = csvread('/Users/sriramreddy/Downloads/ML/2/all_label',0,0);
Z = transpose(X(:,2:end));
Y_b = Y(:,2:end);
K=size(Z);
K=K(1,1);
N=size(Y);
N=N(1,1);
one_n=ones(1,N);
X_b=[Z;one_n];
I_k=eye(K);
zero_k=zeros(K,1);
zero_t=transpose(zeros(K,1));
I_b=[I_k,zero_k;zero_t,0];
X_p=X_b*transpose(X_b);
d=X_b*Y_b;
l_M=[];
VD = csvread('/Users/sriramreddy/Downloads/ML/2/valData.csv',0,0);
VD = transpose(VD(:,2:end));
kill = size(VD);
kill = kill(1,2);
one_n=ones(1,kill);
VD_b=[VD;one_n];
VL = csvread('/Users/sriramreddy/Downloads/ML/2/valLabels.csv',0,0);
VL = VL(:,2:end);
TD = csvread('/Users/sriramreddy/Downloads/ML/2/testData.csv',0,0);
TD = transpose(TD(:,2:end));
kill = size(TD);
kill = kill(1,2);
one_n=ones(1,kill);
TD_b=[TD;one_n];
l_TE=[];
l_CE=[];
l_Kfold=[];
for lambda= [0.785]
%     linspace(0.5,0.7,10)
%     [0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
%     [0.01, 0.1, 1, 10, 100, 1000]
%     lambda=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,1,1.5,2,4,6,8,10, 100, 1000]
%     [0.01,0.05,0.1,0.5,1,2,4,6,8,10, 100, 1000]
% [0.01, 0.1, 1, 10, 100, 1000]
    C=X_p + lambda*I_b;
    Cat=pinv(C);
    w=Cat*d;
    [error]=LOOCV_final(w,Cat,X_b,Y_b);
    l_Kfold=[l_Kfold,error];
%    I can check the error fast.
    Training_error = RMSE(w,X_b,Y_b);
    l_TE=[l_TE,Training_error];
    Crossvalidation_error = RMSE(w,VD_b,VL);
    l_CE=[l_CE,Crossvalidation_error] ;
end
% K=[0.01,0.05,0.1,0.5,1,2,4,6,8,10,100,1000];
K=[0.785];
% linspace(0.5,0.7,10);
% K=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,1,1.5,2,4,6,8,10, 100, 1000];
figure(1);
plot(K,l_TE,'r.-');
xlabel('LAMBDA');
ylabel('TRAINING_ERROR(RED),CV_ERROR(GREEN),LOOCV_RMS_ERROR(BLUE)');
title('ERROR');
hold on;
plot(K,l_CE,'g.-');
hold on;
plot(K,l_Kfold,'b.-');
csvwrite('/Users/sriramreddy/Downloads/ML/2/Output_all_0.785_lambda.csv',transpose(w)*TD_b)
legend('TRAINING_ERROR','CV_ERROR','LOOCV_RMS_ERROR');
% legend('TRAINING_ERROR','CV_ERROR','LOOCV_ERROR');
grid;

