ElemNOD(1:1,1:4)=0;
j=1;
Num=1;
nraz=2*raz_razmax;
for i=1:(length(NODKOORD)-(length(Nod)+nraz))
ElemNOD(Num,1)=Num;
ElemNOD(Num,2)=j;
ElemNOD(Num,3)=j+1;
ElemNOD(Num,4)=j+nraz+2;
ElemNOD(Num,5)=j+nraz+1;
if (mod(j+1,2*raz_razmax+1)==0)
j=j+2;
else
j=j+1;
end
Num=Num+1;
end
clear i j k Num