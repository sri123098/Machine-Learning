classdef HW5_BoW1    
% Practical for Visual Bag-of-Words representation    
% Use SVM, instead of Least-Squares SVM as for MPP_BoW
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 18-Dec-2015
% Last modified: 16-Oct-2018    
    
    methods (Static)
        function main()
            scales = [8, 16, 32, 64];
            normH = 16;
            normW = 16;
            bowCs = HW5_BoW1.learnDictionary(scales, normH, normW);
            
            [trIds, trLbs] = ml_load('/Users/sriramreddy/Downloads/ML/hw5data/bigbangtheory/train.mat',  'imIds', 'lbs');             
            tstIds = ml_load('/Users/sriramreddy/Downloads/ML/hw5data/bigbangtheory/test.mat', 'imIds'); 
            
            trD  = HW5_BoW1.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
            tstD = HW5_BoW1.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Write code for training svm and prediction here            %
%             [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma)
            heart_scale_label = zeros(size(tstIds,2),1);
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
[dataClass, data] = libsvmread('./heart_scale');
%# split into train/test datasets
numTrain = size(trIds,2);
numTest = size(tstIds,2);
%# radial basis function: exp(-gamma*|u-v|^2)
gamma = 2e-3;
% rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);

[trainD testD] = cmpExpX2Kernel(trD, tstD, gamma);
%# compute kernel matrices between every pairs of (train,train) and
%# (test,train) instances and include sample serial number as first column
K =  [ (1:numTrain)', trainD ];
KK = [ (1:numTest)', testD];

%# train and test
model = train(trLbs, K, '-t 4');
[predClass, acc, decVals] = predict(heart_scale_label, KK, model);


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
                
        function bowCs = learnDictionary(scales, normH, normW)
            % Number of random patches to build a visual dictionary
            % Should be around 1 million for a robust result
            % We set to a small number her to speed up the process. 
            nPatch2Sample = 1000000;
            
            % load train ids
            trIds = ml_load('/Users/sriramreddy/Downloads/ML/hw5data//bigbangtheory/train.mat', 'imIds'); 
            nPatchPerImScale = ceil(nPatch2Sample/length(trIds)/length(scales));
                        
            randWins = cell(length(scales), length(trIds)); % to store random patches
            for i=1:length(trIds);
                ml_progressBar(i, length(trIds), 'Randomly sample image patches');
                im = imread(sprintf('/Users/sriramreddy/Downloads/ML/hw5data/bigbangtheory/%06d.jpg', trIds(i)));
                im = double(rgb2gray(im));  
                for j=1:length(scales)
                    scale = scales(j);
                    winSz = [scale, scale];
                    stepSz = winSz/2; % stepSz is set to half the window size here. 
                    
                    % ML_SlideWin is a class for efficient sliding window 
                    swObj = ML_SlideWin(im, winSz, stepSz);
                    
                    % Randomly sample some patches
                    randWins_ji = swObj.getRandomSamples(nPatchPerImScale);
                    
                    % resize all the patches to have a standard size
                    randWins_ji = reshape(randWins_ji, [scale, scale, size(randWins_ji,2)]);                    
                    randWins{j,i} = imresize(randWins_ji, [normH, normW]);
                end
            end
            randWins = cat(3, randWins{:});
            randWins = reshape(randWins, [normH*normW, size(randWins,3)]);
                                    
            fprintf('Learn a visual dictionary using k-means\n');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Use your K-means implementation here                       %
            % to learn visual vocabulary                                 %
            % Input: randWinds contains your data points                 %
            % Output: bowCs: centroids from k-means, one column for each centroid
%             k=1000;
            k=25;
            disp(size(randWins));
%             As given in the question, I have used 1000 clusters.
%             The size of randWins is 256*10000
%  So I'm passing the transpose of the same matrix.
            bowCs = KNN(randWins',k);
%             I'm returning the centroid
%             bowCs contains the centroid such that each column is a
%             centroid for k clusters.
            % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
                
        function D = cmpFeatVecs(imIds, scales, normH, normW, bowCs)
            n = length(imIds);
            D = cell(1, n);
            startT = tic;
            for i=1:n
                ml_progressBar(i, n, 'Computing feature vectors', startT);
                im = imread(sprintf('/Users/sriramreddy/Downloads/ML/hw5data/bigbangtheory/%06d.jpg', imIds(i)));                                
                bowIds = HW5_BoW1.cmpBowIds(im, scales, normH, normW, bowCs);                
                feat = hist(bowIds, 1:size(bowCs,2));
                D{i} = feat(:);
            end
            D = cat(2, D{:});
            D = double(D);
            D = D./repmat(sum(D,1), size(D,1),1);
        end        
        
        % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
        % scales: sizes to densely extract the patches. 
        % normH, normW: normalized height and width oMf patches
        function bowIds = cmpBowIds(im, scales, normH, normW, bowCs)
            im = double(rgb2gray(im));
            bowIds = cell(length(scales),1);
            for j=1:length(scales)
                scale = scales(j);
                winSz = [scale, scale];
                stepSz = winSz/2; % stepSz is set to half the window size here.
                
                % ML_SlideWin is a class for efficient sliding window
                swObj = ML_SlideWin(im, winSz, stepSz);
                nBatch = swObj.getNBatch();
                
                for u=1:nBatch
                    wins = swObj.getBatch(u);
                    
                    % resize all the patches to have a standard size
                    wins = reshape(wins, [scale, scale, size(wins,2)]);                    
                    wins = imresize(wins, [normH, normW]);
                    wins = reshape(wins, [normH*normW, size(wins,3)]);
                    
                    % Get squared distance between windows and centroids
                    dist2 = ml_sqrDist(bowCs, wins); % dist2: k*n matrix, 
                    
                    % bowId: is the index of cluster closest to a patch
                    [~, bowIds{j,u}] = min(dist2, [], 1);                     
                end                
            end
            bowIds = cat(2, bowIds{:});
        end        
    end    
end

