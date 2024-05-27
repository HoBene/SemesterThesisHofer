t_test = load('1dResults/t_test_Sp3_4d_sim.mat'); 
x_test = load('1dResults/x_test_Sp3_4d_sim.mat');
xdot_test = load('1dResults/xdot_test_Sp3_4d_sim.mat');
u_test = load('1dResults/u_test_Sp3_4d_sim.mat');
filename = '1dResults/Im_Mat_4d.csv';  % Specify the filename for the CSV file
options = odeset('RelTol', 100, 'AbsTol', 100);

t_test = t_test.t_test;
t_test = t_test';
xdot_test = xdot_test.xdot_test;
x_test = x_test.x_test;
u_test = u_test.u_test;
u_test_diff = gradient(u_test);

% Define model parameters
f0 = @(x) x;
f1 = @(x,z) exp(-2*abs(x))*z;
f2 = @(x) exp(-2*abs(x))*x^2;
f_dot0 = @(x) cos(x);
f_dot1 = @(x) exp(-abs(x));

F_0=u_test;
F_0_dot=u_test_diff;
tspan = [0 172];
y0 = ones(5,1);
yp0 = ones(5, 1);



[t,y] = ode15i(@(t,y,yp) odefcn(t,y,yp,F_0,F_0_dot,t_test,f0,f1,f2,f_dot0,f_dot1),tspan,y0,yp0,options);

% Plot and save the results
%combined_array = cat(2,x_test, L_sol, t_test');
%writematrix(combined_array, filename, 'Delimiter', ',');
%plot(L_sol(:,1))
%hold on
%plot(x_test(:,2))
%ylim([-4 4])
function dydt = odefcn(t, y,yp, F_0, F_0_dot,t_test, f0, f1, f2, f_dot0, f_dot1)
    F0=interp1(t_test, F_0, t);
    F0_dot=interp1(t_test, F_0_dot, t);
    dydt = zeros(5, 1);
    dydt(1) = yp(1) - f0(y(2));
    dydt(2) = yp(2) - f0(y(3));
    dydt(3) = yp(3) - f0(y(4));

    dydt(4) = yp(4) -1*(5212.904 *f0(yp(2)) -2406.892 *f1(y(1),yp(2)) -39886.698 *f0(yp(2))*f_dot1(yp(1)) + 46901.185 *f1(y(1),yp(2))*f_dot1(yp(1)) + 134050.252 *f0(yp(2))*f_dot1(yp(2)) -142452.476 *f1(y(1),yp(2))*f_dot1(yp(2)) -79859.381 *f0(yp(2))*f_dot1(yp(3)) + 77449.250 *f1(y(1),yp(2))*f_dot1(yp(3)) -21343.438 *f0(yp(2))*f_dot1(yp(4)) + 22289.117 *f1(y(1),yp(2))*f_dot1(yp(4)) + 1001.344 *f0(yp(2))*f_dot1(F0_dot) -1032.274 *f1(y(1),yp(2))*f_dot1(F0_dot) + 551.786 *f0(yp(2))*f_dot0(yp(1)) -598.358 *f1(y(1),yp(2))*f_dot0(yp(1)) + 88.437 *f1(y(1),yp(2))*f_dot0(yp(2)) + 299.138 *f0(yp(2))*f_dot0(yp(3)) -264.711 *f1(y(1),yp(2))*f_dot0(yp(3)));
end

