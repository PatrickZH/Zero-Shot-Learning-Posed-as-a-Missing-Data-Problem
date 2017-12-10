% This is the code of the method 'MDP' in paper
% 'Zero-Shot Learning Posed as a Missing Data Problem'
% Author: Bo Zhao
% Email: bozhao@pku.edu.cn
% Date: 2017.12.09

% If you use the code or data, you can cite our paper:
% @inproceedings{zhao2017zero,
%   title={Zero-Shot Learning Posed as a Missing Data Problem},
%   author={Zhao, Bo and Wu, Botong and Wu, Tianfu and Wang, Yizhou},
%   booktitle={Proceedings of the IEEE International Conference on Computer Vision},
%   pages={2616--2622},
%   year={2017}
% }


clc;clear;
load('CUB_ImageFeatures_ResGoog.mat');
load('CUB_WordVectors.mat');
load('CUB_splits_default.mat');
load('CUB_Attributes.mat');
dim_f = 400;
[ImageFeatures, mapping] = compute_mapping(ImageFeatures, 'PCA', dim_f);
dim_w = size(WordVectors,2);
dim_a = size(Attribute,2);
CN = 200;


%% Normalization
% ImageFeatures = ImageFeatures./max(max(ImageFeatures));
Attribute = Attribute./max(max(Attribute));
% WordVectors = WordVectors./max(max(WordVectors));s)));

accs_A_Rec = [];
accs_A_EM_mu = [];
accs_W_Rec = [];
accs_W_EM_mu = [];
accs_AW_Rec = [];
accs_AW_EM_mu = [];
accs_AW_EM_mu_2 = [];
accs_AW_Rec_C2 = [];
accs_AW_EM_mu_C2 = [];


for iter = 1
    tstart = clock();
    disp(iter);

    %% data preparation
    % seen/unseen (train/test) split
    list_all = [1:CN]';
    list_test = splits(iter,:)';
    list_train = list_all;
    list_train(list_test) = [];
    
    FeaTrain = []; % cluster centers (mean vectors or prototypes) of all training seen classes 
    WorTrain = []; % word vectors of all training seen classes
    AttTrain = []; % attribute vectors of all training seen classes
    for i = 1:length(list_train)
        index = find(Labels==list_train(i));
        FeaTrain = [FeaTrain;mean(ImageFeatures(index,:))];
        WorTrain = [WorTrain;WordVectors(list_train(i),:)];
        AttTrain = [AttTrain;Attribute(list_train(i),:)];
    end
    FeaTest = [];
    WorTest = [];
    AttTest = [];
    X = [];
    Y = [];
    for i = 1:length(list_test)
        index = find(Labels==list_test(i));
        X = [X;ImageFeatures(index,:)];
        Y = [Y;Labels(index)];
        FeaTest = [FeaTest;mean(ImageFeatures(index,:))];
        WorTest = [WorTest;WordVectors(list_test(i),:)];
        AttTest = [AttTest;Attribute(list_test(i),:)];
    end
    
    
    % reconstruction cofficient of Word vectors
    sc_w = zeros(size(WorTrain,1), size(WorTest,1));
    for k = 1 : size(WorTest,1)
        [sc_w(:, k)] = LeastR(WorTrain', WorTest(k, :)', 0.5); % you can tune the L1 coefficient
    end
    FeaRecon_W = (FeaTrain'*sc_w)';
    
    % reconstruction cofficient of Attributes
    sc_a = zeros(size(AttTrain,1), size(AttTest,1));
    for k = 1 : size(AttTest,1)
        [sc_a(:, k)] = LeastR(AttTrain', AttTest(k, :)', 0.5); % you can tune the L1 coefficient
    end
    FeaRecon_A = (FeaTrain'*sc_a)';
    
    FeaRecon_AW = (FeaRecon_W+FeaRecon_A)./2;
    
    % reconstruction cofficient of Attributes + Word vectors
    AWTrain = [AttTrain,WorTrain];
    AWTest = [AttTest,WorTest];
    sc_aw = zeros(size(AWTrain,1), size(AWTest,1));
    for k = 1 : size(AWTest,1)
        [sc_aw(:, k)] = LeastR(AWTrain', AWTest(k, :)', 0.5);
    end
    FeaRecon_AW_C2 = (FeaTrain'*sc_aw)';
    
    
    %% direct Reconstructed Centers classification
    [accuracy_A_Rco,Labels_predict_R] = classifier_nearest(X,FeaRecon_A,list_test,Y);
    disp(['accuracy_A_Rco = ',num2str(accuracy_A_Rco)]);
    accs_A_Rec = [accs_A_Rec;accuracy_A_Rco];
    
    [accuracy_W_Rco,Labels_predict_R] = classifier_nearest(X,FeaRecon_W,list_test,Y);
    disp(['accuracy_W_Rco = ',num2str(accuracy_W_Rco)]);
    accs_W_Rec = [accs_W_Rec;accuracy_W_Rco];
    
    [accuracy_AW_Rco_C2,Labels_predict_R] = classifier_nearest(X,FeaRecon_AW_C2,list_test,Y);
    disp(['accuracy_AW_Rco_C2 = ',num2str(accuracy_AW_Rco_C2)]);
    accs_AW_Rec_C2 = [accs_AW_Rec_C2;accuracy_AW_Rco_C2];
    
    %% Reconstructed Centers + kmeans
    opts = statset('Display','off');
    [Idx,C_W,sumD_W]=kmeans(X,length(list_test),'Start',FeaRecon_W,'Options',opts);
    [Idx,C_A,sumD_A]=kmeans(X,length(list_test),'Start',FeaRecon_A,'Options',opts); 
    [Idx,C2,sumD]=kmeans(X,length(list_test),'Start',FeaRecon_AW_C2,'Options',opts); % new
    
    [accuracy_A_EM_mu,Labels_predict] = classifier_nearest(X,C_A,list_test,Y);
    disp(['accuracy_A_EM_mu = ',num2str(accuracy_A_EM_mu)]);
    accs_A_EM_mu = [accs_A_EM_mu;accuracy_A_EM_mu];
    
    [accuracy_W_EM_mu,Labels_predict] = classifier_nearest(X,C_W,list_test,Y);
    disp(['accuracy_W_EM_mu = ',num2str(accuracy_W_EM_mu)]);
    accs_W_EM_mu = [accs_W_EM_mu;accuracy_W_EM_mu];
   
    [accuracy_AW_EM_mu_C2,Labels_predict] = classifier_nearest(X,C2,list_test,Y);
    disp(['accuracy_AW_EM_mu_C2 = ',num2str(accuracy_AW_EM_mu_C2)]);
    accs_AW_EM_mu_C2 = [accs_AW_EM_mu_C2;accuracy_AW_EM_mu_C2];
    
    tend = clock();
    fprintf('Each iter costs time = %f\n',etime(tend,tstart));
    fprintf('*******************************\n');
    
end

fprintf('accuracy_Rco means Acc. of Syn.-Cen. \naccuracy_EM_mu means Acc. of GMM-EM-U \n ');

