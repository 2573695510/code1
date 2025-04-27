
%% 回归算法NRBO-LSTM-Attention  
%% 清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 导入数据
%res = xlsread('修正风速后的nwp+风电场实测数据.xlsx','A2:I300');
res = xlsread('洗.xlsx');

%%  数据分析
num_size = 0.7;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
%res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

disp('程序计算量大，运行较慢请耐心等待！');

%%  创建待优化函数
ObjFcn = @CostFunction;
%% 优化参数设置
SearchAgents = 6; % 种群数量
Max_iterations = 10 ; % 迭代次数
lowerbound = [1e-6,1e-5,4 ];%三个参数的下限
upperbound = [1e-2,1e-1,100 ];%三个参数的上限
dimension = 3;%数量，即要优化的LSTM参数个数

%% 优化LSTM
[Best_score,Best_pos,Convergence_curve]=NRBO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,ObjFcn);

%%  得到最优参数
NumOfUnits       = round( Best_pos(1,3));       % 最佳隐藏层节点数
InitialLearnRate = Best_pos(1,2);               % 最佳初始学习率
L2Regularization = Best_pos(1,1);               % 最佳L2正则化系数

%%  创建网络，
layers = [ ...
    sequenceInputLayer(size(P_train,1))              % 输入层,即输入的特征变量个数
    lstmLayer(NumOfUnits)                            % lstm层
    selfAttentionLayer(1,2)                          % 创建一个单头，2个键和查询通道的自注意力层 
    reluLayer                                        % Relu激活层
    fullyConnectedLayer(outdim)                      % 全连接层
    regressionLayer];                                % 回归层


%% 参数设置
options = trainingOptions('adam', ... % 优化算法Adam
'MaxEpochs', 400, ... % 最大训练次数
'GradientThreshold', 1, ... % 梯度阈值
'InitialLearnRate', InitialLearnRate, ... % 初始学习率
'LearnRateSchedule', 'piecewise', ... % 学习率调整
'LearnRateDropPeriod', 350, ... % 训练850次后开始调整学习率
'LearnRateDropFactor',0.2, ... % 学习率调整因子
'L2Regularization', L2Regularization, ... % 正则化参数
'ExecutionEnvironment', 'cpu',... % 训练环境
'Verbose', 0, ... % 关闭优化过程
'Plots', 'training-progress'); % 画出曲线

%%  训练
net = trainNetwork(vp_train, vt_train, layers, options);

%%  预测
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  数据格式转换
T_sim1 = cell2mat(T_sim1);  %转为普通数组
T_sim2 = cell2mat(T_sim2);
T_sim1 = double(T_sim1');   %输出转置才是列的答案，double一下精度
T_sim2 = double(T_sim2');

%%  绘图
%%  适应度曲线
figure
plot(Convergence_curve,'b-', 'LineWidth', 1.5);
title('NRBO-LSTM-Attention', 'FontSize', 10);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度值', 'FontSize', 10);
grid off
set(gcf,'color','w')

%% 测试集结果
figure;
plotregression(T_test,T_sim2,'回归图');
set(gcf,'color','w')
figure;
ploterrhist(T_test-T_sim2,'误差直方图');
set(gcf,'color','w')

%%  均方根误差 RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%  决定系数
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%%  均方误差 MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;

%%  RPD 剩余预测残差
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;
SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;

%% 平均绝对误差MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));

%% 平均绝对百分比误差MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

%%  训练集绘图
figure
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1.5)
legend('真实值','NRBO-LSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
set(gcf,'color','w')

%% 预测集绘图
figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1.5)
legend('真实值','NRBO-LSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
set(gcf,'color','w')

%% 测试集误差图
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('测试集样本编号')

ylabel('预测误差')

title('测试集预测误差')
grid on;
legend('NRBO-LSTM预测输出误差')
set(gcf,'color','w')

%% 绘制线性拟合图
%% 训练集拟合效果图
figure
plot(T_train,T_sim1,'*r');
xlabel('真实值')
ylabel('预测值')
string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 预测集拟合效果图
figure
plot(T_test,T_sim2,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'测试集效果图';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 求平均
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% 总数据线性预测拟合图
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'所有样本拟合预测图';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')

%% 打印出评价指标
disp('-----------------------误差计算--------------------------')
disp('预测集的评价结果如下所示：')
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSEP为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])
grid

%%

%xlswrite('预测功率7.14.xlsx',tsim);




