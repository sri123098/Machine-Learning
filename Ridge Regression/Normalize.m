function [Check] = Normalize(Z)
    Size = size(Z);
    N=Size(1,2);
    p=linspace(1,N,N);
    Mean=[mean(Z(:,p))];
    Deviation=[std(Z(:,p))];
    Check=(Z-Mean)./Deviation;
end