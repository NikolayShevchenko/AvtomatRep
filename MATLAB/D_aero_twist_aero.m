if f_0~=f_k
for j=1:1/raz_chord+1
for i=1:nraz+1
z = abs(NODKOORD(nraz*(j-1)+j-1+i,3)-2*Lk);
f_tw = ((f_k-f_0)/(Lk)^a_t)*z^a_t+f_0;
if f_0~=0
y_twisted = NODKOORD(nraz*(j-1)+j-1+i,4)*f_tw/f_0;
else
y_twisted = NODKOORD(nraz*(j-1)+j-1+i,4)*f_tw/f_k;
end
ATWIST(nraz*(j-1)+j-1+i,1) = y_twisted;
end
end
NODKOORD(:,4) = ATWIST(:);
clear j i z ATWIST y_twisted f_tw z
end