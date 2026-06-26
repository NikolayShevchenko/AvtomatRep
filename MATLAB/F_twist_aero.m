%
a_k1 = a_k*pi/180;
axis = NODKOORD(1:nraz+1,2:4);
b_z1v = axis-NODKOORD((nraz+1)*(1/raz_chord)+1:(nraz+1)*(1/raz_chord+1),2:4);
for m=1:length(b_z1v)
b_z(m,1) = sqrt(b_z1v(m,1)^2+b_z1v(m,2)^2+b_z1v(m,3)^2);
end
delta_z_k = b_z(1)*sin(a_k1);
delta_x_k = b_z(1)*(1-cos(a_k1));
for j=1:1/raz_chord+1
for i=1:nraz+1
z = abs(NODKOORD(nraz*(j-1)+j-1+i,3)-2*Lk);
delta_z = (delta_z_k/(Lk)^n_t)*z^n_t;
delta_x = (delta_x_k/(Lk)^n_t)*z^n_t;
L = NODKOORD(nraz*(j-1)+j-1+i,2:4)- axis(i,:);
b = norm(L)/b_z(i);
TWIST(nraz*(j-1)+j-1+i,1)= abs(b*delta_x);
TWIST(nraz*(j-1)+j-1+i,2)= 0;
TWIST(nraz*(j-1)+j-1+i,3)= -abs(b*delta_z);
end
end
NODKOORDTWIST(:,1)=NODKOORD(:,1);
NODKOORDTWIST(:,2)=NODKOORD(:,2)+TWIST(:,1);
NODKOORDTWIST(:,3)=NODKOORD(:,3)+TWIST(:,2);
NODKOORDTWIST(:,4)=NODKOORD(:,4)+TWIST(:,3);
NODKOORD=NODKOORDTWIST;
axis2 = NODKOORDTWIST(1:nraz+1,2:4);
b_z2v = axis2-NODKOORDTWIST((nraz+1)*(1/raz_chord)+1:(nraz+1)*(1/raz_chord+1),2:4);
for t=1:length(b_z2v)
b_z2(t,1) = sqrt(b_z2v(t,1)^2+b_z2v(t,2)^2+b_z2v(t,3)^2);
end
for f=1:length(b_z2v)
angle(f,1) = acosd((b_z1v(f,1)*b_z2v(f,1)+b_z1v(f,2)*b_z2v(f,2)+b_z1v(f,3)*b_z2v(f,3))/(b_z(f)*b_z2(f)));
end
for u=1:nraz
angletwist(u,1) = 0.5*(angle(u,1)+angle(u+1,1));
end
check = b_z-b_z2;
for aj = 1:nraz/2
z_otn(aj,1) = abs((axis(aj,2)-2*Lk))/Lk;
z_otn(nraz-aj+1,1) = z_otn(aj,1);
end
for q=1:nraz
chorddis(q,1) = 0.5*(b_z(q)+b_z(q+1));
end
TWISTDistr1(:,1) = (2*Lk-axis(1:nraz/2+1,2))./Lk;
TWISTDistr1(:,2) = TWIST((nraz+1)*(1/raz_chord)+1:(nraz+1)*(1/raz_chord)+1+nraz/2,1);
TWISTDistr1(:,3) = TWIST((nraz+1)*(1/raz_chord)+1:(nraz+1)*(1/raz_chord)+1+nraz/2,2);
TWISTDistr1(:,4) = TWIST((nraz+1)*(1/raz_chord)+1:(nraz+1)*(1/raz_chord)+1+nraz/2,3);
TWISTDistr1(:,5) = angle(1:1+nraz/2,1);
TWISTDistr1(:,6) = check(1:1+nraz/2,1);
TWISTDistr = flipud(TWISTDistr1);
ERRORTWIST(1,1) = max(check)*100/bk;
clear angletwist ERRORTWIST TWIST b_z1v aj q b_z u i j f t m k L n b_z1 delta_z_k delta_x_k delta_x delta_z L a_k1 z b alpha_i TWISTDistr1 axis2 b_z2 b_z2v check k angle NODKOORDTWIST TWISTDistr