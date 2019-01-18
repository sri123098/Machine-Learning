%1)Generate the initial data for the training and testing.
%2)Iterate the loop
%3)     Train & Remove the non support vectors for the negative examples from the training set.
%4)     Once it is done, iterate over the train images
%5)     Use detect function to get the patches
%6)     Get the rectangles, use them for images, compute the features and then normalize the features
%7)     Add them to the training data and then continue

%Logical Indexing is an imp concept
%Another indexing variation, logical indexing, has proven to be both useful and expressive. In logical indexing, you use a single, logical array for the matrix subscript. MATLAB extracts the matrix elements corresponding to the nonzero values of the logical array. The output is always in the form of a column vector. For example, A(A > 12) extracts all the elements of A that are greater than 12.
%&& is used for short circuit operator in matlab
% w = warning ('on','all');
% imReg = im(neg_rects{i}(2,j):neg_rects{i}(4,j), neg_rects{i}(1,j):neg_rects{i}(3,j),:);
% warning(w);
% id=w.identifier;
% warning('off',id);

Ap=[];
Obj=[];
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
for iter=1:7
    count=0;
    C=10;
    [w,b,obj_min_dual]=SVM_final10(trD,trLb,C);
    %From the training, I have to remove the -ve examples which are not satisfying the 
    [d,n]=size(trD);
    svm=[];
    for i=1:n
        if (trLb(i)==-1) && (trLb(i)*(w'*trD(:,i) + b) <= 1)
            svm=[svm i];
        end
    end
    trD(:,~svm)=[];
    %Removing all the non support vectors from the training data.
    trLb(~svm)=[];
    Neg_tr=[];
    %Removing all the non support vectors from the training labels.
    %Training data does not contain A.I need to include the hardest examples.
    
    for i=1:93
        im = imread(sprintf('/Users/sriramreddy/Downloads/ML/hw4data/%sIms/%04d.jpg', "train", i));
        randrects{i}=HW4_Utils.detect(im,w,b,0);
        %fifth column is the score. Pick the ones with negative as these are negative examples from the image patches.
         sum_neg= sum(randrects{i}(end,:) <0);
%          neg_rects{i}=randrects{i}(:,end-sum_neg+1: end);
%          neg_rects{i} =randrects{i}(:, 1:end-sum_neg);
         neg_rects{i}=randrects{i}(:, randrects{i}(5,:)>=-1);
%       Above one is basically positive Rectangles  +13   
        [imH, imW,~] = size(im);
         %The following has been referred from the helper function getPosAndRandomNegHelper
         % remove random rects{i} that do not lie within image boundaries
        badIdxs = or(neg_rects{i}(3,:) > imW, neg_rects{i}(4,:) > imH);
        neg_rects{i} = neg_rects{i}(:,~badIdxs);
         % Remove random rects{i} that overlap more than 30% with an annotated upper body
        load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "train"), 'ubAnno');
        ubs = ubAnno{i};
        kill = size(neg_rects{i},2);
%          if sum_neg < kill
             for j=1:size(ubs,2)
                 overlap = HW4_Utils.rectOverlap(neg_rects{i}, ubs(:,j));                    
                 neg_rects{i} = neg_rects{i}(:, overlap < 0.33);
                 if isempty(neg_rects{i})
                     break;
                 end
             end
%          end
         [D_i, R_i] = deal(cell(1, size(neg_rects{i},2)));
%          Creation of the data structure.
         % Generate feature vectors
        for j=1:size(neg_rects{i}, 2)
            neg_rects{i}(1:4,j)=round(neg_rects{i}(1:4,j));
            imReg = im(neg_rects{i}(2,j):neg_rects{i}(4,j), neg_rects{i}(1,j):neg_rects{i}(3,j),:);
            imReg = imresize(imReg, HW4_Utils.normImSz);
            D_i{j} = HW4_Utils.cmpFeat(rgb2gray(imReg));
            R_i{j} = imReg;
        end
        mat_form = cell2mat(D_i);
        size(mat_form);   
        mat_form=HW4_Utils.l2Norm(double(mat_form));
        Neg_tr = [Neg_tr mat_form];
        %If it is not working horzcat
        count =  count + size(mat_form, 2);
        if count > 1200
            disp("count");
            disp(count);
            break;
        end
    end
    trD=[trD Neg_tr];
    labels = size(Neg_tr,2);
    trLb = [trLb;-1*ones(labels,1)];
    HW4_Utils.genRsltFile(w, b, "val", "r32");
    [ap, prec, rec] = HW4_Utils.cmpAP("r32", "val");
    Ap = [Ap ap]
    disp("accuracy and obj");
    %disp(ap)
    %disp(-obj_min_dual);
    Obj = [Obj -obj_min_dual]
end
% Ac;
% Obj;
% dplot(Ac);
% plot(Obj);     

%
%       % Helper function to get training data for training upper body classifier
%        % Positive data is feature vectors for annotated upper bodies
%        % Negative data is feature vectors for random image patches
%        % Inputs:
%        %   dataset: either 'train' or 'val'
%        % Outputs:
%        %   D: d*n data matrix, each column is a HOG feature vector
%        %   lb: n*1 label vector, entries are 1 or -1        
%        %   imRegs: 64*64*3*n array for n images
%        function [D, lb, imRegs] = getPosAndRandomNegHelper(dataset)
%            rng(1234); % reset random generator. Keep same seed for repeatability
%            load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, dataset), 'ubAnno');
%            [posD, negD, posRegs, negRegs] = deal(cell(1, length(ubAnno)));            
%            
%            for i=1:length(ubAnno)
%                ml_progressBar(i, length(ubAnno), 'Processing image');
%
%                im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, dataset, i));
%                %im = rgb2gray(im);
%                ubs = ubAnno{i}; % annotated upper body
%                if ~isempty(ubs)
%                    [D_i, R_i] = deal(cell(1, size(ubs,2)));
%                    for j=1:length(D_i)
%                        %no of face---- upper body faces
%                        ub = ubs(:,j);
%                        imReg = im(ub(2):ub(4), ub(1):ub(3),:);
%                        imReg = imresize(imReg, HW4_Utils.normImSz);
%                        D_i{j} = HW4_Utils.cmpFeat(rgb2gray(imReg));
%                        R_i{j} = imReg;
%                    end 
%                    posD{i}    = cat(2, D_i{:});                    
%                    posRegs{i} = cat(4, R_i{:});
%                end
%
%                
%                
%                % sample k random patches; some will be used as negative exampels
%                % Choose k sufficiently large to ensure success
%                k = 1000;
%                [imH, imW,~] = size(im);
%                randLeft = randi(imW, [1, k]);
%                randTop = randi(imH, [1, k]);
%                randSz = randi(min(imH, imW), [1, k]);
%                randrects{i} = [randLeft; randTop; randLeft + randSz - 1; randTop + randSz - 1];
%                
%                % remove random rects{i} that do not lie within image boundaries
%                badIdxs = or(randrects{i}(3,:) > imW, randrects{i}(4,:) > imH);
%                randrects{i} = randrects{i}(:,~badIdxs);
%                
%                % Remove random rects{i} that overlap more than 30% with an annotated upper body
%                for j=1:size(ubs,2)
%                    overlap = HW4_Utils.rectOverlap(randrects{i}, ubs(:,j));                    
%                    randrects{i} = randrects{i}(:, overlap < 0.3);
%                    if isempty(randrects{i})
%                        break;
%                    end;
%                end;
%                
%                % Now extract features for some few random patches
%                nNeg2SamplePerIm = 2;
%                [D_i, R_i] = deal(cell(1, nNeg2SamplePerIm));
%                for j=1:nNeg2SamplePerIm
%                    imReg = im(randrects{i}(2,j):randrects{i}(4,j), randrects{i}(1,j):randrects{i}(3,j),:);
%                    imReg = imresize(imReg, HW4_Utils.normImSz);
%                    R_i{j} = imReg;
%                    D_i{j} = HW4_Utils.cmpFeat(rgb2gray(imReg));                    
%                end
%                negD{i} = cat(2, D_i{:});                
%                negRegs{i} = cat(4, R_i{:});
%            end    
%            posD = cat(2, posD{:});
%            negD = cat(2, negD{:});   
%            D = cat(2, posD, negD);
%            lb = [ones(size(posD,2),1); -ones(size(negD,2), 1)];
%            imRegs = cat(4, posRegs{:}, negRegs{:});            
%        end;           
%      
