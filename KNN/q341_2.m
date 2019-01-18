%  model =train(trLbs, trD','-v 5')
% I have loaded the trLbs and trD in my kernel and then used the above line
% for getting the cross validation accuracy. 

i=1;
model=zeros(6,6);
% 0.0001, 0.001, 0.01, 0.1,
% 0.0001, 0.001, 0.01, 0.1
for C=[ 1, 10, 100, 1000, 10000, 100000]
    for gamma=[ 1, 10, 100, 1000, 10000, 100000]
%model(i,j) = train(trLbs, K, sprintf('-c %f -v %d', C, 5));j=j+1;
        model(i) = train(trLbs,trD', sprintf('-g %f -c %f -v %d -q',gamma,C, 5));
        i=i+1;
    end
%i=i+1;
end