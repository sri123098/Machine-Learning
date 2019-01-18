scales = [8, 16, 32, 64];
normH = 16;
normW = 16;
bowC1s = HW5_BoW1.learnDictionary(scales, normH, normW);
[trId1s, trLb1s] = ml_load('/Users/sriramreddy/Downloads/ML/hw5data/bigbangtheory/train.mat',  'imIds', 'lbs');             
tstId1s = ml_load('/Users/sriramreddy/Downloads/ML/hw5data/bigbangtheory/test.mat', 'imIds'); 
trD1  = HW5_BoW1.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
tstD1 = HW5_BoW1.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
csvwrite('/Users/sriramreddy/Downloads/ML/hw5data/trD1.csv',trD);
csvwrite('/Users/sriramreddy/Downloads/ML/hw5data/tstD1.csv',tstD);


% heart_scale_label = zeros(size(tstIds,2),1);     
% [dataClass, data] = libsvmread('./heart_scale');
% numTrain = size(trIds,2);
% numTest = size(tstIds,2);
% gamma = 2e-3;
% [trainD testD] = cmpExpX2Kernel(trD', tstD', gamma);
% disp(size(trainD));
% disp(size(testD));
% K =  [ (1:numTrain)' trainD' ];
% KK = [ (1:numTest)' testD'];
% model = train(trLbs, K, '-c 2048 -t 4 -v 5');
% [predClass, acc, decVals] = predict(heart_scale_label, KK, model);



% C=0.1 g 0.001
% [labels,data] = libsvmread('./heart_scale');
%# grid of parameters
% folds = 5;
% [C,gamma] = meshgrid(2:0.5:13, -1:0.5:7);
% %# grid search, and cross-validation
% cv_acc = zeros(numel(C),1);
% for i=1:numel(C)
%     cv_acc(i) = train(trLbs, trD',sprintf('-c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
% end
% disp("Accuracy");
% disp(cv_acc);
% %# pair (C,gamma) with best accuracy
% [~,idx] = max(cv_acc);
% best_C = 2^C(idx);
% best_gamma = 2^gamma(idx);
% disp(best_C);
% disp(best_gamma);
% %# contour plot of paramter selection
% contour(C, gamma, reshape(cv_acc,size(C))), colorbar
% hold on
% plot(C(idx), gamma(idx), 'rx')
% text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
%     'HorizontalAlign','left', 'VerticalAlign','top')
% hold off
% xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')
%# now you can train you model using best_C and best_gamma

%# read dataset



%model = train(trLbs, trD','-c 2048 -g 2.82842712');
%[predict_label, accuracy, dec_values] = predict(heart_scale_label, tstD', model);
%csvwrite('/Users/sriramreddy/Downloads/ML/hw5data/sample.csv',predict_label);


%             [, 'libsvm_options']            
%          -training_label_vector:
%             An m by 1 vector of training labels (type must be double).
%         -training_instance_matrix:
%             An m by n matrix of m training instances with n features.
%             It can be dense or sparse (type must be double).
%         -libsvm_options:
%             A string of training options in the same format as that of LIBSVM.
% >> load heart_scale
% >> model = svmtrain(heart_scale_label,heart_scale_inst,'-c 1 -g 0.07 -v 5',);
% *
% optimization finished, #iter = 134
% nu = 0.433785
% obj = -101.855060, rho = 0.426412
% nSV = 130, nBSV = 107
% Total nSV = 130
% >> [predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model);
% Accuracy = 86.6667% (234/270) (classification) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
     