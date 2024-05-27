t_test = load('1dResults/t_test_Sp3_4d_sim.mat');   % load data 
u_test = load('1dResults/u_test_Sp3_4d_sim.mat');
x_test = load('1dResults/x_test_Sp3_4d_sim.mat');
xdot_test = load('1dResults/xdot_test_Sp3_4d_sim.mat');

x_test = x_test.x_test;
xdot_test = xdot_test.xdot_test;

u_test_diff = gradient(u_test.u_test);      

t_test = t_test.t_test';
F_0 = u_test.u_test;


options = odeset('RelTol', 1e-4, 'AbsTol', 1e-9, 'MaxStep', 0.01);  % adjust solver options

filename = '1dResults/Sp3_4d_Sim.csv';  % Specify the filename for the CSV file
filename1 = '1dResults/Sp3_4d_Sim_test.csv';  % Specify the filename for the CSV file


tspan = linspace(0, 203, 280);
y0 = [x_test(1,1) x_test(1,4) x_test(1,7) x_test(1,10) x_test(1,2) x_test(1,5) x_test(1,8) x_test(1,11) x_test(1,3) x_test(1,6) x_test(1,9) x_test(1,12)];% x_test(1,5) x_test(1,6) x_test(1,7) x_test(1,8)]
tspan = double(tspan);
y0 = double(y0);

f0 = @(x) x;
f1 = @(x,z) exp(-2*abs(x))*z;
f2 = @(x) exp(-2*abs(x))*cos(x)*x;
f3 = @(x) exp(-2*abs(x))*x^2;
f4 = @(x) sin(x)*x; 

[t, y] = ode23s(@(t, y) odefunc(t, y, t_test, F_0, f0, f1, f2, f3, f4), tspan, y0,options);
%t=t(1:100:end);
%y=y(1:100:end,:);
t_test=t_test(1:10:end);
x_test=x_test(1:10:end,:);
combined_array = cat(2,t_test,x_test);
combined_array1 = cat(2,t,y);
plot(t(:),y(:,1))
hold on
plot(t_test(:),x_test(:,1))
%writematrix(combined_array, filename, 'Delimiter', ',');
%writematrix(combined_array1, filename1, 'Delimiter', ',');

function dydt = odefunc(t, y, t_test, F_0, f0, f1, f2, f3, f4)
    F0 = interp1(t_test, F_0, t);
    dydt = zeros(12, 1);
    dydt(1) = y(2);
    dydt(2) = y(3);
    dydt(3) = y(4);
    dydt(4) =    -2.700 *sin(1 *y(1)) -0.505 *cos(1 *y(1)) + 0.492 *sin(1 *y(5)) + 1.409 *cos(1 *y(9)) -0.554 *sin(1 *y(6)) -0.846 *cos(1 *y(6)) -0.327 *cos(1 *y(10)) -0.380 *cos(1 *y(3)) -0.331 *cos(1 *y(4)) + 0.562 *sin(2 *y(1)) -0.437 *sin(2 *y(5)) -0.327 *cos(2 *y(3)) -2.605 *y(1) + 1.511 *y(5) -5.813 *y(9) -0.930 *y(2) + 8.497 *y(6) + 2.481 *y(10) -7.467 *y(3) -2.092 *y(7) -1.022 *y(11) -0.407 *y(4) + 0.363 *y(8) ;
    dydt(5) = y(6);
    dydt(6) = y(7);
    dydt(7) = y(8);
    dydt(8) =  1.452 *cos(1 *y(1)) + 2.061 *sin(1 *y(9)) + 0.342 *sin(1 *y(2)) -0.375 *sin(1 *y(3)) + 0.275 *sin(1 *F0) -0.768 *cos(1 *F0) -0.497 *sin(2 *y(1)) + 0.710 *sin(2 *y(5)) + 5.146 *y(1) -6.207 *y(5) + 4.886 *y(9) -1.663 *y(2) -1.600 *y(6) -7.411 *y(7) ;
    dydt(9) = y(10);
    dydt(10) = y(11);
    dydt(11) = y(12);
    dydt(12) = 0.405 *sin(1 *y(1)) -0.660 *cos(1 *y(1)) + 0.696 *sin(1 *y(5)) + 1.266 *cos(1 *y(5)) -2.728 *sin(1 *y(9)) + 0.690 *cos(1 *y(9)) -0.649 *sin(1 *y(2)) + 1.697 *sin(1 *y(10)) + 0.524 *cos(1 *y(10)) + 0.341 *cos(1 *y(7)) -0.554 *cos(1 *y(11)) + 0.198 *sin(1 *F0) -1.779 *cos(1 *F0) -0.358 *cos(2 *y(1)) -0.412 *cos(2 *y(9)) + 0.343 *cos(2 *y(2)) -0.321 *cos(2 *y(6)) -0.356 *cos(2 *y(10)) + 0.262 *sin(2 *F0) + 0.577 *cos(2 *F0) -9.592 *y(1) + 6.697 *y(5) -6.784 *y(9) -2.289 *y(2) -7.416 *y(6) + 1.748 *y(10) -0.644 *y(3) + 0.909 *y(7) -7.075 *y(11) -0.607 *y(8) + 0.270 *y(12) -0.763 *F0  ;
end