a_sound=331.3*sqrt(1+T/273.15);
M=Vpol/a_sound;
beta = 1/sqrt(1-M^2); %Поправка на сжимаемость

for j=1:length(ElemNOD(:,1))
    xa=NODDOPL(j,1)*10^-3;
    ya=NODDOPL(j,2)*10^-3;
    za=NODDOPL(j,3)*10^-3;
    xb=NODDOPP(j,1)*10^-3;
    yb=NODDOPP(j,2)*10^-3;
    zb=NODDOPP(j,3)*10^-3;
    Fx=0*(zb-za)-Vpol*degtorad(alpha)*(yb-ya);
    Fy=-Vpol*(zb-za);
    Fz=Vpol*(yb-ya)-0*(xb-xa);
    F_t=sqrt(Fx^2+Fy^2+Fz^2);
    F_vector(j,1)= Fx/F_t;
    F_vector(j,2)= Fy/F_t;
    F_vector(j,3)= Fz/F_t;
end

b=transpose([-Vpol 0 -Vpol*degtorad(alpha)]*transpose(NormalP));
x=(As\b)*beta;

for i=1:length(x)
    x_t(i,1)=x(i,1);
    delta_Y(i,1)=x_t(i,1)*Vpol*plotn*shag*10^-3;
    delta_Y_x(i,1)=delta_Y(i,1)*F_vector(i,1);
    delta_Y_y(i,1)=delta_Y(i,1)*F_vector(i,2);
    delta_Y_z(i,1)=delta_Y(i,1)*F_vector(i,3);
end

xx=length(x)/(1/raz_chord);
j=1;
k=1;
for i=1:length(x)
    Gamma(k,j)=x_t(i);
    detdetY(k,j)=delta_Y(i);
    detdetY_x(k,j)=delta_Y_x(i);
    detdetY_y(k,j)=delta_Y_y(i);
    detdetY_z(k,j)=delta_Y_z(i);
    if (mod(i,xx)==0)
        k=1;
        j=j+1;
    else
        k=k+1;
    end
end

for i=1:raz_razmax*2
    detY(i,1)=sum(detdetY_z(i,:));
end

bcax = 0.5*(b0+bk);
Y=sum(detY(:));
Cy=Y/(0.5*plotn*Vpol^2*S*10^-6);

for i=1:nraz
    circulation(i,1)=detY(i,1)*nraz/abs(Y);
end

wind = As_drag*x_t;
for i=1:raz_razmax*2
    detGamma(i,1)=(sum(Gamma(i,:)));
end

for i=1:length(wind)
    det_drag(i,1) =0.5*plotn*wind(i,1)*detGamma(i,1)*shag*10^-3;
end

Xi = -sum(det_drag);
Cxi=Xi/(0.5*plotn*Vpol^2*S*10^-6);
lz_ef=1/(pi*(Cxi/Cy^2));

%Вязкая составляющая
Re=Vpol*(bcax*10^-3)/visc;
c_cp=(ck_+c_)*0.5;
n_c=1+2*c_cp+9*c_cp^2;
n_m=1/sqrt(1+0.2*M^2)*(1+5*c_cp^2*M);
Cx0=0.087*n_c*n_m/(log10(Re)-1.6)^2;

%Суммарная сила
Cx=Cx0+Cxi;
Ki = Cy/Cxi;
K_kr = Cy/(Cxi+Cx0);

for i=1:length(detY)/2
    Q(i,1)=sum(detY(1:i));
    dL(i,1)=((length(detY)/2)-i+1)/(0.5*length(detY))*Lk-shag*0.5;
end

for i=1:length(detY)/2
    M_bend(i,1)=sum(detY(1:i,1).*dL(1:i,1))*10^-3;
end

for i=1:length(NODDDOW)
    r_fl(i,1)=abs(NODDDOW(i,1)-min(NODKOORD(:,2)))*10^-3;
    delta_pitch1(i,1)=r_fl(i)*delta_Y(i);
end

xx=length(x)/(1/raz_chord);
j=1;
k=1;
for i=1:length(x)
    delta_pitch(k,j)=delta_pitch1(i);
    if (mod(i,xx)==0)
        k=1;
        j=j+1;
    else
        k=k+1;
    end
end

delta_pitch2=sum(delta_pitch,2);
pitch_m1=delta_pitch2(1:nraz/2);
for i=1:length(pitch_m1)
    pitch_m(i,1)=-sum(pitch_m1(1:i));
end

M_z = 2*(pitch_m(length(pitch_m)));
C_mz= M_z/(0.5*plotn*Vpol^2*S*bcax*10^-9);
X_d = -C_mz/Cy;

%Производные
alpha2=alpha-1;
alpha3=alpha+1;
b3=transpose([-Vpol 0 -Vpol*degtorad(alpha3)]*transpose(NormalP));
Gamma3=(As\b3)*beta;
b2=transpose([-Vpol 0 -Vpol*degtorad(alpha2)]*transpose(NormalP));
Gamma2=(As\b2)*beta;

for i=1:length(Gamma2)
    delta_Y2(i,1)=Gamma2(i,1)*F_vector(i,3)*Vpol*plotn*shag*10^-3;
end

for i=1:length(Gamma3)
    delta_Y3(i,1)=Gamma3(i,1)*F_vector(i,3)*Vpol*plotn*shag*10^-3;
end

Y3=sum(delta_Y3(:));
Y2=sum(delta_Y2(:));
Cy2=Y2/(0.5*plotn*Vpol^2*S*10^-6);
Cy3=Y3/(0.5*plotn*Vpol^2*S*10^-6);
Cy_alpha = (Cy3-Cy2)/(degtorad(alpha3)-degtorad(alpha2));
alpha0 = radtodeg(degtorad(alpha)-Cy/Cy_alpha);

for i=1:length(NODDDOW)
    delta_pitch_1(i,1)=r_fl(i)*delta_Y2(i);
end

xx=length(x)/(1/raz_chord);
j=1;
k=1;
for i=1:length(x)
    delta_pitch_(k,j)=delta_pitch_1(i);
    if (mod(i,xx)==0)
        k=1;
        j=j+1;
    else
        k=k+1;
    end
end

delta_pitch_2=sum(delta_pitch_,2);
pitch_m1_=delta_pitch_2(1:nraz/2);
for i=1:length(pitch_m1_)
    pitch_m_(i,1)=-sum(pitch_m1_(1:i));
end

M_z2 = 2*(pitch_m_(length(pitch_m_)));
C_mz2= M_z2/(0.5*plotn*Vpol^2*S*bcax*10^-9);
Cmz_alpha = (C_mz2-C_mz)/(degtorad(alpha2)-degtorad(alpha));
X_F = -Cmz_alpha/Cy_alpha;

% Очистка только временных переменных, созданных внутри I_solver
% (все важные переменные сохраняются для повторных вызовов)
clear a_sound M beta xa ya za xb yb zb Fx Fy Fz F_t F_vector ...
    b x x_t delta_Y_x delta_Y_y xx j k ...
    Gamma detdetY detdetY_x detdetY_y detdetY_z detY bcax Y ...
    circulation wind detGamma det_drag Xi lz_ef Re n_c n_m ...
    K_kr Q dL M_bend r_fl delta_pitch1 delta_pitch ...
    delta_pitch2 pitch_m1 pitch_m M_z C_mz X_d alpha2 alpha3 ...
    b3 Gamma3 b2 Gamma2 delta_Y2 delta_Y3 Y2 Y3 Cy2 Cy3 Cy_alpha ...
    alpha0 delta_pitch_1 delta_pitch_ delta_pitch_2 pitch_m1_ ...
    pitch_m_ M_z2 C_mz2 Cmz_alpha X_F