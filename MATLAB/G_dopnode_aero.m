NODDOPL(1:1,1:3)=0;
NODDOPP(1:1,1:3)=0;
NODDDOW(1:1,1:3)=0; %%
%NODKOORD=NODKOORDTWIST;
for i=1:length(ElemNOD(:,1))
nod1=ElemNOD(i,2);
nod2=ElemNOD(i,5);
nod3=ElemNOD(i,3);
nod4=ElemNOD(i,4);
%%
x1=NODKOORD(nod1,2);
x2=NODKOORD(nod2,2);
x3=NODKOORD(nod3,2);
x4=NODKOORD(nod4,2);
z1=NODKOORD(nod1,4);
z2=NODKOORD(nod2,4);
z3=NODKOORD(nod3,4);
z4=NODKOORD(nod4,4);
%%%%%%%%%%%%%%%
b=x1-x2;
x14=x2+b/4;
b1=x3-x4;
x141=x4+b1/4;
%%%%%%%%%%%%%%%
a=(abs(z2)-abs(z1));
z14=z2+a*0.25;
a1=(abs(z4)-abs(z3));
z141=z4+a1*0.25;
z21(i)=abs(abs(z2+a*0.75)-abs(z4+a1*0.75));
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
NODDOPL(i,1)=x14;
NODDOPL(i,2)=NODKOORD(nod1,3);
NODDOPL(i,3)=z14;
ElemNOD(i,6)=i;
%
NODDOPP(i,1)=x141;
NODDOPP(i,2)=NODKOORD(nod3,3);
NODDOPP(i,3)=z141;
ElemNOD(i,7)=i;
%
if ((NODKOORD(nod1,2)-NODKOORD(nod3,2))>0)
NODDDOW(i,1)=NODKOORD(nod1,2)-0.5*abs(NODKOORD(nod1,2)-NODKOORD(nod3,2))-(b+b1)/8;
else
NODDDOW(i,1)=NODKOORD(nod3,2)-0.5*abs(NODKOORD(nod1,2)-NODKOORD(nod3,2))-(b+b1)/8;
end
NODDDOW(i,2)=NODKOORD(nod3,3)-(NODKOORD(nod3,3)-NODKOORD(nod1,3))/2;
if (z2>z1)
z_1=(z2-z1)*0.25+z1;
else
z_1=(z1-z2)*0.25+z2;
end
if (z4>z3)
z_2=(z4-z3)*0.25+z3;
else
z_2=(z3-z4)*0.25+z4;
end
if (z_2>z_1)
z_=(z_2+z_1)*0.5;
else
z_=(z_1+z_2)*0.5;
end
NODDDOW(i,3)=z_;
ElemNOD(i,8)=i;
end
for i=1:2*raz_razmax
NODDDOW_dop(i,1)=sl*Lk;
NODDDOW_dop(i,2)=NODDDOW(i,2);
NODDDOW_dop(i,3)=(NODKOORD(i+1,4)+NODKOORD(i,4))/2;;
end
clear z_1 z_2 z_ z21 x1 y1 x2 y2 x3 y3 x4 y4 z1 z2 z3 z4 z14 z141 x14 x141 a a1 b b1 a1 nod1 nod2 nod3 nod4 i