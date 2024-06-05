% 输入数据
X = [0, 34, 67, 101, 135, 202, 259, 336, 404, 471, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259];
Y = [196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 0, 24, 49, 73, 98, 147, 196, 245, 294, 342, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196];
Z = [372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 0, 47, 93, 140, 186, 279, 372, 465, 558, 651];
T = [15.18, 21.36, 25.72, 32.29, 34.03, 39.45, 43.15, 43.46, 40.83, 30.75, 33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73, 18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22];

% 转换为适合网络的输入格式
input = [X; Y; Z];
output = T;

% 创建一个具有10个隐藏层神经元的前馈神经网络
hiddenLayerSize = 5;
net = fitnet(hiddenLayerSize);
net.performParam.regularization = 0.1;


% 分割数据为训练集、验证集和测试集
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% 训练网络
[net, tr] = train(net, input, output);

% 预测输出
output_pred = net(input);

% 计算性能
performance = perform(net, output, output_pred);

% 可视化结果
figure;
plot(output, 'b', 'DisplayName', 'Actual');
hold on;
plot(output_pred, 'r', 'DisplayName', 'Predicted');
legend('show');
xlabel('Sample Index');
ylabel('T');
title('Actual vs Predicted');
% 计算均方误差
mse = mean((output - output_pred).^2);

% 显示性能指标
disp(['Mean Squared Error: ', num2str(mse)]);
disp(['Performance: ', num2str(performance)]);


% 定义目标函数（用于优化）
objective_function = @(xyz) -net(xyz');  % 由于遗传算法是求最小值，所以我们取负值

% 定义输入变量的上下界
lb = [0, 0, 0];   % 下界（可以根据实际情况调整）
ub = [471, 342, 651]; % 上界（可以根据实际情况调整）

% 使用遗传算法进行优化
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50);
[x_opt, fval_opt] = ga(objective_function, 3, [], [], [], [], lb, ub, [], options);

% 最优解
max_value = -fval_opt; % 记得取负值还原
max_X = x_opt(1);
max_Y = x_opt(2);
max_Z = x_opt(3);

% 显示最大值及其对应的输入数据点
disp(['Global Maximum value: ', num2str(max_value)]);
disp(['At input (X, Y, Z): (', num2str(max_X), ', ', num2str(max_Y), ', ', num2str(max_Z), ')']);