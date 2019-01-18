% This is the solution for 3.1
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg;
% Loading the data 
C=10;
[w,b] = SVM_hardmining(C,trD,valD,trLb,valLb);
HW4_Utils.genRsltFile(w, b, "val", "r31");
[ap, prec, rec] = HW4_Utils.cmpAP("r31", "val");
disp(ap);
% The results are stored in r31



