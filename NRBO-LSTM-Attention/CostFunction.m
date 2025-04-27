function valError = CostFunction(optVars)
%% 赋值
L2Regularization =abs(optVars(1)); % 正则化参数
InitialLearnRate=abs(optVars(2)); % 初始学习率
NumOfUnits = abs(round(optVars(3))); % 神经元个数
%%  获取数据
vp_train = evalin('base', 'vp_train');
vt_train = evalin('base', 'vt_train');

%%  输入和输出特征个数
inputSize    = size(vp_train{1}, 1);
numResponses = size(vt_train{1}, 1);

%%  设置网络结构
opt.layers = [ ...
    sequenceInputLayer(inputSize)       % 输入层

    lstmLayer(NumOfUnits)               % lstm层
    selfAttentionLayer(1,2)             % 创建一个单头，2个键和查询通道的自注意力层 
    reluLayer                           % Relu激活层

    fullyConnectedLayer(numResponses)   % 回归层

    regressionLayer];

%%  设置网络参数
opt.options = trainingOptions('adam', ...             % 优化算法Adam
    'MaxEpochs', 100, ...                             % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', InitialLearnRate, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 70, ...                    % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', L2Regularization, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'none');                                 % 不画出曲线

%%  训练网络
net = trainNetwork(vp_train, vt_train, opt.layers, opt.options);

%%  得到网络预测值
t_sim1 = predict(net, vp_train); 

%%  计算误差
valError = sqrt(double(mse(vt_train, t_sim1)));

end