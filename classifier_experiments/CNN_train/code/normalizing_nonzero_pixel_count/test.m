%possible movements
C = combnk([-1,0,1,-1,0,1],2);
C1 = unique(C, 'rows');
zeroInd = all(C1==0,2);
C1(zeroInd,:) = [];

disp(C);
disp(C1);
