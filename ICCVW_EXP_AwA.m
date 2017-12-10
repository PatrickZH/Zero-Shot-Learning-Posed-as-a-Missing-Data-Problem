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

%% AwA
load('AwA_ImageFeatures_VGG.mat'); % 4096 dim
dim_f = 80;
[ImageFeatures, mapping] = compute_mapping(ImageFeatures, 'PCA', dim_f);
load('AwA_WordVectors.mat');
dim_w = size(WordVectors,2);
load('AwA_Attributes.mat');
Attribute = attributes_embedding_c; % use continuous-value attributes
dim_a = size(Attribute,2);
CN = 50;

%% Normalization
% ImageFeatures = ImageFeatures./max(max(ImageFeatures));
Attribute = Attribute./max(max(Attribute));
% WordVectors = WordVectors./max(max(WordVectors));



accs_A_Rec = [];
accs_A_EM_mu = [];
accs_A_EM_sigma = [];
accs_W_Rec = [];
accs_W_EM_mu = [];
accs_W_EM_sigma = [];
accs_AW_Rec = [];
accs_AW_EM_mu = [];
accs_AW_EM_sigma = [];


load('AwA_splits_default.mat');
tn = 0;
for iter = 1
    tstart = clock();
    disp(iter);
    
    %% data preparation
    % seen/unseen (train/test) split
    list_all = [1:CN]';
    list_test = splits(iter,:)';
    list_train = list_all;
    list_train(list_test) = [];
    disp(list_test');
    
    FeaTrain = []; % cluster centers (mean vectors or prototypes) of all training seen classes 
    WorTrain = []; % word vectors of all training seen classes
    AttTrain = []; % attribute vectors of all training seen classes
    X_Train = [];
    Y_Train = [];
    for i = 1:length(list_train)
        index = find(Labels==list_train(i));
        X_Train = [X_Train;ImageFeatures(index,:)];
        Y_Train = [Y_Train;Labels(index,:)];
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
    
    % reconstruction cofficients of Attributes
    sc_a = zeros(size(AttTrain,1), size(AttTest,1));
    for k = 1 : size(AttTest,1)
        [sc_a(:, k)] = LeastR(AttTrain', AttTest(k, :)', 0.5); % you can tune the L1 coefficient
    end
    FeaRecon_A = (FeaTrain'*sc_a)';
    
    % reconstruction cofficients of Attributes + Word vectors
    AWTrain = [AttTrain,WorTrain];
    AWTest = [AttTest,WorTest];
    sc_aw = zeros(size(AWTrain,1), size(AWTest,1));
    for k = 1 : size(AWTest,1)
        [sc_aw(:, k)] = LeastR(AWTrain', AWTest(k, :)', 0.5);
    end
    FeaRecon_AW2 = (FeaTrain'*sc_aw)';
    
    %% GMM for seen classes   
    fprintf('Computing GM parameters...\n');
    Ip = reshape( eye(dim_f),1,dim_f*dim_f);
    
    mu = []; % centers of all train classes
    Sigma = [];
    ComponentProportion = [];
    for i = 1:length(list_train)
        % fprintf('Computing %d th GM parameters...\n',i);
        index = find(Labels==list_train(i));
        X_each = ImageFeatures(index,:);
        gm = fitgmdist(X_each,1, 'CovarianceType','diagonal');
        mu = [mu; gm.mu];
        Sigma = [Sigma; reshape(gm.Sigma,1,dim_f)];%reshape(Ip,1,dim_f*dim_f)];%
        ComponentProportion = [ComponentProportion; gm.ComponentProportion];
    end
    
    mu_Rec_W = (mu'*sc_w)';
    mu_Rec_A = (mu'*sc_a)';
    
    Sigma_Rec_W = (Sigma'*sc_w)';
    Sigma_Rec_A = (Sigma'*sc_a)';
    
    mu_Rec_AW_C2 = (mu'*sc_aw)';
    Sigma_Rec_AW_C2 = (Sigma'*sc_aw)';
    
    
    
    %% A
    fprintf('---------------A----------------------------\n');
    S.mu = (mu_Rec_A);
    [accuracy_A_Rec,Labels_predict] = classifier_nearest(X,S.mu,list_test,Y);
    fprintf('accuracy_Rec = %f\n',accuracy_A_Rec);
    accs_A_Rec = [accs_A_Rec;accuracy_A_Rec];
    
    Sigma_Rec = (Sigma_Rec_A);
    Temp = Sigma_Rec; 
    Sigma_Rec = [];
    for i = 1:length(list_test)
        Sigma_Rec = [Sigma_Rec, Temp(i,:)];
    end
    
    temp = Sigma_Rec;
    temp(temp<=0) = [];
    Sigma_Rec(Sigma_Rec<=0) = min(min(temp));
    RegV = 0.01;
    S.Sigma = reshape(Sigma_Rec,1,dim_f,length(list_test)) + repmat(RegV*ones(1,dim_f),[1,1,length(list_test)]); %initSigma
    S.ComponentProportion = ones(1,length(list_test))/length(list_test);
    
    gm = fitgmdist(X,length(list_test),'Start',S , 'CovarianceType','diagonal','RegularizationValue',0.01);
    [accuracy_A_EM_mu,Labels_predict] = classifier_nearest(X,gm.mu,list_test,Y);
    fprintf('accuracy_Gaussian_Mu = %f\n',accuracy_A_EM_mu);
    accs_A_EM_mu = [accs_A_EM_mu;accuracy_A_EM_mu];
    
    mu_all = gm.mu;
    sigma_all = gm.Sigma;
    Px = Cal_GMM_ZH(X,mu_all,sigma_all);
    [v,Pred_ind] = max(Px,[],2);
    Pred_label_GMM = list_test(Pred_ind);
    accuracy_A_EM_sigma = sum(Pred_label_GMM==Y)/length(Y);
    fprintf('accuracy_Gaussian_Sig = %f\n',accuracy_A_EM_sigma);
    accs_A_EM_sigma = [accs_A_EM_sigma;accuracy_A_EM_sigma];
    
    %% W
    fprintf('---------------W----------------------------\n');
    S.mu = (mu_Rec_W);
    [accuracy_W_Rec,Labels_predict] = classifier_nearest(X,S.mu,list_test,Y);
    fprintf('accuracy_Rec = %f\n',accuracy_W_Rec);
    accs_W_Rec = [accs_W_Rec;accuracy_W_Rec];
    
    Sigma_Rec = (Sigma_Rec_W);
    Temp = Sigma_Rec; 
    Sigma_Rec = [];
    for i = 1:length(list_test)
        Sigma_Rec = [Sigma_Rec, Temp(i,:)];
    end
    
    temp = Sigma_Rec;
    temp(temp<=0) = [];
    Sigma_Rec(Sigma_Rec<=0) = min(min(temp));
    RegV = 0.01;
    S.Sigma = reshape(Sigma_Rec,1,dim_f,length(list_test)) + repmat(RegV*ones(1,dim_f),[1,1,length(list_test)]); 
    S.ComponentProportion = ones(1,length(list_test))/length(list_test);
    
    gm = fitgmdist(X,length(list_test),'Start',S , 'CovarianceType','diagonal','RegularizationValue',0.01);
    [accuracy_W_EM_mu,Labels_predict] = classifier_nearest(X,gm.mu,list_test,Y);
    fprintf('accuracy_Gaussian_Mu = %f\n',accuracy_W_EM_mu);
    accs_W_EM_mu = [accs_W_EM_mu;accuracy_W_EM_mu];
    
    mu_all = gm.mu;
    sigma_all = gm.Sigma;
    Px = Cal_GMM_ZH(X,mu_all,sigma_all);
    [v,Pred_ind] = max(Px,[],2);
    Pred_label_GMM = list_test(Pred_ind);
    accuracy_W_EM_sigma = sum(Pred_label_GMM==Y)/length(Y);
    fprintf('accuracy_Gaussian_Sig = %f\n',accuracy_W_EM_sigma);
    accs_W_EM_sigma = [accs_W_EM_sigma;accuracy_W_EM_sigma];
    
    %% A+W
    fprintf('---------------AW----------------------------\n');
    S.mu = (mu_Rec_W+mu_Rec_A)./2;
    [accuracy_AW_Rec,Labels_predict] = classifier_nearest(X,S.mu,list_test,Y);
    fprintf('accuracy_Rec = %f\n',accuracy_AW_Rec);
    accs_AW_Rec = [accs_AW_Rec;accuracy_AW_Rec];
    
    Sigma_Rec = (Sigma_Rec_W+Sigma_Rec_A)./2;
    Temp = Sigma_Rec;
    Sigma_Rec = [];
    for i = 1:length(list_test)
        Sigma_Rec = [Sigma_Rec, Temp(i,:)];
    end
    
    temp = Sigma_Rec;
    temp(temp<=0) = [];
    Sigma_Rec(Sigma_Rec<=0) = min(min(temp));
    RegV = 0.01;
    S.Sigma = reshape(Sigma_Rec,1,dim_f,length(list_test)) + repmat(RegV*ones(1,dim_f),[1,1,length(list_test)]);
    S.ComponentProportion = ones(1,length(list_test))/length(list_test);
    
    gm = fitgmdist(X,length(list_test),'Start',S , 'CovarianceType','diagonal','RegularizationValue',0.01);
    [accuracy_AW_EM_mu,Labels_predict] = classifier_nearest(X,gm.mu,list_test,Y);
    fprintf('accuracy_Gaussian_Mu = %f\n',accuracy_AW_EM_mu);
    accs_AW_EM_mu = [accs_AW_EM_mu;accuracy_AW_EM_mu];
    
    mu_all = gm.mu;
    sigma_all = gm.Sigma;
    Px = Cal_GMM_ZH(X,mu_all,sigma_all);
    [v,Pred_ind] = max(Px,[],2);
    Pred_label_GMM = list_test(Pred_ind);
    accuracy_AW_EM_sigma = sum(Pred_label_GMM==Y)/length(Y);
    fprintf('accuracy_Gaussian_Sig = %f\n',accuracy_AW_EM_sigma);
    accs_AW_EM_sigma = [accs_AW_EM_sigma;accuracy_AW_EM_sigma];
    
    tend = clock();
    fprintf('Each iter costs time = %f\n',etime(tend,tstart));
    fprintf('*******************************\n');
end

fprintf('accuracy_Rec means Acc. of Syn.-Cen. \naccuracy_EM_mu means Acc. of GMM-EM-U \naccuracy_EM_sigma means Acc. of GMM-EM-D\n');





