function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
%       trainK=chiSquared(trainD,trainD,gamma);
%       testK=chiSquared(testD,trainD,gamma);  
      trainK=chi(trainD,trainD,gamma);
      testK=chi(testD,trainD,gamma); 
%      for i=1:n
%      for j=1:n
%           rbf=(-sum((trD(:,i)-trD(:,j)).^2)/gamma); 
%           H(i,j) = trLb(i)*trLb(j)*rbf; 
%      end
%      end
end



