i=1;
% i corresponds to C
j=1;
% j corresponds to gamma
model=zeros(6,6);
for C=[ 1, 10, 100, 1000, 10000, 100000]
     for gamma=[ 1, 10, 100, 1000, 10000, 100000]
% for C=[ 500]
%     for gamma=[60]
        [trainD testD] = cmpExpX2Kernel(trD, tstD, gamma);
        numTest=size(testD,1);
        numTrain=size(trainD,1);
        disp(size(trainD));
        disp(size(testD));
        K =  [ (1:numTrain)' trainD ];
        KK = [ (1:numTest)' testD];
% model(i,j) = train(trLbs, K, sprintf('-c %f -v %d', C, 5));j=j+1;
        model(i) = train(trLbs, K, sprintf('-c %f -t 4 -q -v %d', C,5));
%         [predClass, acc, decVals] = predict(heart_scale_label, KK, model);
%         csvwrite('/Users/sriramreddy/Downloads/ML/hw5data/sample.csv',predClass);
%         model3(i) = train(trLbs, K, sprintf('-c %f -v %d -q -t 4', C, 5));
        i=i+1;
    end
% i=i+1;
end