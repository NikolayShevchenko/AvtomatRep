k=1;
d=0;
dd=shag*(raz_razmax);
b1=bk;
b2=b0;
for i=1:length(Nod)
for j=1:2*raz_razmax+1
if (j<=raz_razmax)
NOD(k,1)=Nod(i,1)*b1+dd*tan(deg2rad(str));
NOD(k,2)=-dd+2*Lk;
NOD(k,3)=Nod(i,2)*b1;
dd=dd-shag;
else
NOD(k,1)=Nod(i,1)*b2+d*tan(deg2rad(str));
NOD(k,2)=d+2*Lk;
NOD(k,3)=Nod(i,2)*b2;
d=d+shag;
b2=b2-(b0-bk)/raz_razmax;
end
k=k+1;
b1=b1+(b0-bk)/raz_razmax;
end
d=0;
dd=shag*(raz_razmax);
b1=bk;
b2=b0;
end
for i=1:length(NOD)
NODKOORD(i,1)=i;
NODKOORD(i,2)=NOD(i,1);
NODKOORD(i,3)=NOD(i,2);
NODKOORD(i,4)=NOD(i,3);
end
clear d i j k b1 b2 dd