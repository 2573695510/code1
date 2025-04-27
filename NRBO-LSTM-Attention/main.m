
%% �ع��㷨NRBO-LSTM-Attention  
%% ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%% ��������
%res = xlsread('�������ٺ��nwp+��糡ʵ������.xlsx','A2:I300');
res = xlsread('ϴ.xlsx');

%%  ���ݷ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
outdim = 1;                                  % ���һ��Ϊ���
num_samples = size(res, 1);                  % ��������
%res = res(randperm(num_samples), :);         % �������ݼ�����ϣ������ʱ��ע�͸��У�
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��

%%  ����ѵ�����Ͳ��Լ�
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

disp('��������������н��������ĵȴ���');

%%  �������Ż�����
ObjFcn = @CostFunction;
%% �Ż���������
SearchAgents = 6; % ��Ⱥ����
Max_iterations = 10 ; % ��������
lowerbound = [1e-6,1e-5,4 ];%��������������
upperbound = [1e-2,1e-1,100 ];%��������������
dimension = 3;%��������Ҫ�Ż���LSTM��������

%% �Ż�LSTM
[Best_score,Best_pos,Convergence_curve]=NRBO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,ObjFcn);

%%  �õ����Ų���
NumOfUnits       = round( Best_pos(1,3));       % ������ز�ڵ���
InitialLearnRate = Best_pos(1,2);               % ��ѳ�ʼѧϰ��
L2Regularization = Best_pos(1,1);               % ���L2����ϵ��

%%  �������磬
layers = [ ...
    sequenceInputLayer(size(P_train,1))              % �����,�������������������
    lstmLayer(NumOfUnits)                            % lstm��
    selfAttentionLayer(1,2)                          % ����һ����ͷ��2�����Ͳ�ѯͨ������ע������ 
    reluLayer                                        % Relu�����
    fullyConnectedLayer(outdim)                      % ȫ���Ӳ�
    regressionLayer];                                % �ع��


%% ��������
options = trainingOptions('adam', ... % �Ż��㷨Adam
'MaxEpochs', 400, ... % ���ѵ������
'GradientThreshold', 1, ... % �ݶ���ֵ
'InitialLearnRate', InitialLearnRate, ... % ��ʼѧϰ��
'LearnRateSchedule', 'piecewise', ... % ѧϰ�ʵ���
'LearnRateDropPeriod', 350, ... % ѵ��850�κ�ʼ����ѧϰ��
'LearnRateDropFactor',0.2, ... % ѧϰ�ʵ�������
'L2Regularization', L2Regularization, ... % ���򻯲���
'ExecutionEnvironment', 'cpu',... % ѵ������
'Verbose', 0, ... % �ر��Ż�����
'Plots', 'training-progress'); % ��������

%%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);

%%  Ԥ��
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���ݸ�ʽת��
T_sim1 = cell2mat(T_sim1);  %תΪ��ͨ����
T_sim2 = cell2mat(T_sim2);
T_sim1 = double(T_sim1');   %���ת�ò����еĴ𰸣�doubleһ�¾���
T_sim2 = double(T_sim2');

%%  ��ͼ
%%  ��Ӧ������
figure
plot(Convergence_curve,'b-', 'LineWidth', 1.5);
title('NRBO-LSTM-Attention', 'FontSize', 10);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��ֵ', 'FontSize', 10);
grid off
set(gcf,'color','w')

%% ���Լ����
figure;
plotregression(T_test,T_sim2,'�ع�ͼ');
set(gcf,'color','w')
figure;
ploterrhist(T_test-T_sim2,'���ֱ��ͼ');
set(gcf,'color','w')

%%  ��������� RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%  ����ϵ��
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%%  ������� MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;

%%  RPD ʣ��Ԥ��в�
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;
SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;

%% ƽ���������MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));

%% ƽ�����԰ٷֱ����MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

%%  ѵ������ͼ
figure
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1.5)
legend('��ʵֵ','NRBO-LSTM-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'ѵ����Ԥ�����Ա�';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
set(gcf,'color','w')

%% Ԥ�⼯��ͼ
figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1.5)
legend('��ʵֵ','NRBO-LSTM-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'���Լ�Ԥ�����Ա�';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
set(gcf,'color','w')

%% ���Լ����ͼ
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('���Լ��������')

ylabel('Ԥ�����')

title('���Լ�Ԥ�����')
grid on;
legend('NRBO-LSTMԤ��������')
set(gcf,'color','w')

%% �����������ͼ
%% ѵ�������Ч��ͼ
figure
plot(T_train,T_sim1,'*r');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string = {'ѵ����Ч��ͼ';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% Ԥ�⼯���Ч��ͼ
figure
plot(T_test,T_sim2,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'���Լ�Ч��ͼ';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% ��ƽ��
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% ����������Ԥ�����ͼ
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'�����������Ԥ��ͼ';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')

%% ��ӡ������ָ��
disp('-----------------------������--------------------------')
disp('Ԥ�⼯�����۽��������ʾ��')
disp(['ƽ���������MAEΪ��',num2str(MAE2)])
disp(['�������MSEΪ��       ',num2str(mse2)])
disp(['���������RMSEPΪ��  ',num2str(error2)])
disp(['����ϵ��R^2Ϊ��  ',num2str(R2)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD2)])
disp(['ƽ�����԰ٷֱ����MAPEΪ��  ',num2str(MAPE2)])
grid

%%

%xlswrite('Ԥ�⹦��7.14.xlsx',tsim);




