k=1;
n=1;
for i=1:length(ElemNOD(:,1))
x1=NODKOORD(ElemNOD(i,2),2);
y1=NODKOORD(ElemNOD(i,2),3);
z1=NODKOORD(ElemNOD(i,2),4);
x2=NODKOORD(ElemNOD(i,3),2);
y2=NODKOORD(ElemNOD(i,3),3);
z2=NODKOORD(ElemNOD(i,3),4);
x3=NODKOORD(ElemNOD(i,4),2);
y3=NODKOORD(ElemNOD(i,4),3);
z3=NODKOORD(ElemNOD(i,4),4);
x4=NODKOORD(ElemNOD(i,5),2);
y4=NODKOORD(ElemNOD(i,5),3);
z4=NODKOORD(ElemNOD(i,5),4);
V1=[x3-x1 y3-y1 z3-z1];
V2=[x2-x4 y2-y4 z2-z4];
n=cross(V2,V1);
area_cell(k,1) = norm(n);
NormalP(k,1:3)=n/sqrt(n(1,1)^2+n(1,2)^2+n(1,3)^2);
k=k+1;
end
STWIST = sum(area_cell);
ERRORTWIST(2,1) = abs(S-STWIST)*100/S;
clear V1 V2 i n x0 y0 z0 x1 y1 z1 x2 y2 z2 k area_cell k ERRORTWIST y3 z3 x3 y4 z4 x4