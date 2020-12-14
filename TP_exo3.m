%%%%%%%%%%%%%%%%%%%%%%% Kismet_data_intent %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract the prosodic features for Kismet (f0 and energy).
f0k_files=dir('./Kismet_data_intent/*.f0');
enk_files=dir('./Kismet_data_intent/*.en');

f0k_database=[];
en_database_k=[];

database_target_k=[];
for i=1:length(f0k_files)
    f0k_file_name=strcat(f0k_files(i).folder,'/',f0k_files(i).name);
    
    if contains(f0k_files(i).name,'ap')
        class='ap'; %class approval
    elseif contains(f0k_files(i).name,'pw')
        class='pw'; %class prohibition
    elseif contains(f0k_files(i).name,'at')
        class='at'; %class attention
    end
    
    if strcmp(class,'ap') || strcmp(class,'pw')|| strcmp(class,'at') %If there is ap, at or pw in the name of the file
        database_target_k=[database_target_k;class]; 

        %Functionals for f0 (fundamental frequency)
        f0k_sample=readtable(f0k_file_name,'FileType','text','ReadRowNames',0,'Delimiter','space');
        f0k_sample=f0k_sample{:,:};

        local_derivative_functional_k=(f0k_sample(2:end,2)-f0k_sample(1:end-1,2))./(f0k_sample(2:end,1)-f0k_sample(1:end-1,1));
        % filter f0=0 : f0_sample(:,2)==0 (remove it) but we need to process the mean absolute of
        % local derivative before
        %Filter f0==0
        f0k_diff_from_zero_mask=(f0k_sample(:,2)~=0);
        f0k_sample=f0k_sample(f0k_diff_from_zero_mask,:);
        mean_absolute_local_derivative_functional_k=mean(abs(local_derivative_functional_k(f0k_diff_from_zero_mask(1:end-1))));

        mean_functional_k=mean(f0k_sample(:,2));
        max_functional_k=max(f0k_sample(:,2));
        range_functional_k=max(f0k_sample(:,2))-min(f0k_sample(:,2));
        variance_functional_k=var(f0k_sample(:,2));
        median_functional_k=median(f0k_sample(:,2));
        first_quartile_functional_k=quantile(f0k_sample(:,2),0.25);
        third_quartile_functional_k=quantile(f0k_sample(:,2),0.75);

        f0k_database=[f0k_database;mean_functional_k,max_functional_k,range_functional_k,variance_functional_k,median_functional_k,first_quartile_functional_k,third_quartile_functional_k,mean_absolute_local_derivative_functional_k];

        
        % Functionals for en (energy)
        enk_file_name=strcat(f0k_file_name(1:end-2),"en");
        enk_sample=readtable(enk_file_name,'FileType','text','ReadRowNames',0,'Delimiter','space');
        enk_sample=enk_sample{:,:};

        local_derivative_functional_k=(enk_sample(2:end,2)-enk_sample(1:end-1,2))./(enk_sample(2:end,1)-enk_sample(1:end-1,1));

        enk_diff_from_zero_mask=(f0k_sample(:,2)~=0);
        enk_sample=enk_sample(f0k_diff_from_zero_mask,:);

        mean_absolute_local_derivative_k=mean(abs(local_derivative_functional_k(f0k_diff_from_zero_mask(1:end-1))));

        mean_functional_k=mean(enk_sample(:,2));
        max_functional_k=max(enk_sample(:,2));
        range_functional_k=max(enk_sample(:,2))-min(enk_sample(:,2));
        variance_functional_k=var(enk_sample(:,2));
        median_functional_k=median(enk_sample(:,2));
        first_quartile_functional_k=quantile(enk_sample(:,2),0.25);
        third_quartile_functional_k=quantile(enk_sample(:,2),0.75);
        en_database_k=[en_database_k;mean_functional_k,max_functional_k,range_functional_k,variance_functional_k,median_functional_k,first_quartile_functional_k,third_quartile_functional_k,mean_absolute_local_derivative_functional_k];
    end
end

database_k=[f0k_database,en_database_k]; % X of the training database
% Y -class database_target

nb_neighbors_k = 1; % modify it and test the results

train_test_ratio_k = 0.6;

[mk,nk] = size(database_k) ;
idx_k = randperm(mk) ;
training_database_k = database_k(idx_k(1:round(train_test_ratio_k*mk)),:) ; 
testing_database_k = database_k(idx_k(round(train_test_ratio_k*mk)+1:end),:) ;
training_database_target_k = database_target_k(idx_k(1:round(train_test_ratio_k*mk)),:);
testing_database_target_k = database_target_k(idx_k(round(train_test_ratio_k*mk)+1:end),:);

model_k=fitcknn(training_database_k,training_database_target_k,'NumNeighbors',nb_neighbors_k);

yfit_k = model_k.predict(testing_database_k);

% Compute confusion matrix table (2 by 2)
confusion_matrix_k=confusionmat(testing_database_target_k,yfit_k);
disp("Confusion matrice of Kismet");
disp(confusion_matrix_k)

% Accuracy (number of yfit==testing_database_targets / nb of samples (size
% of testing_database_targets)
accuracy_k=sum(diag(confusion_matrix_k))/sum(sum(confusion_matrix_k));
%accuracy_Kismet = ['Accuracy of Kismet_data_intent = ',accuracy_k];
disp("Accuracy of Kismet");
disp(accuracy_k);


%%%%%%%%%%%%%%%%%%%%%%%%%% Baby_Ears %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract the prosodic features for Baby Ears (f0 and energy).
f0_files=dir('./BabyEars_data/*.f0');
en_files=dir('./BabyEars_data/*.en');

f0_database=[]; % Nx8 matrix with N the number of f0 files
en_database=[]; % Nx8 matrix with N the number of en files

database_target=[];
for i=1:length(f0_files)
    f0_file_name=strcat(f0_files(i).folder,'/',f0_files(i).name);
    
    if contains(f0_files(i).name,'ap')
        class='ap';
    elseif contains(f0_files(i).name,'pr')
        class='pw';
    elseif contains(f0_files(i).name,'at')
        class='at';
    end
    
    if strcmp(class,'ap') || strcmp(class,'pw')|| strcmp(class,'at')
        database_target=[database_target;class];

        f0_sample=readtable(f0_file_name,'FileType','text','ReadRowNames',0,'Delimiter','space');
        f0_sample=f0_sample{:,:};

        local_derivative_functional=(f0_sample(2:end,2)-f0_sample(1:end-1,2))./(f0_sample(2:end,1)-f0_sample(1:end-1,1));
        % filter f0=0 : f0_sample(:,2)==0 (remove it) but we need to process the mean absolute of
        % local derivative before
        %Filter f0==0
        f0_diff_from_zero_mask=(f0_sample(:,2)~=0);
        f0_sample=f0_sample(f0_diff_from_zero_mask,:);
        mean_absolute_local_derivative_functional=mean(abs(local_derivative_functional(f0_diff_from_zero_mask(1:end-1))));

        mean_functional=mean(f0_sample(:,2));
        max_functional=max(f0_sample(:,2));
        range_functional=max(f0_sample(:,2))-min(f0_sample(:,2));
        variance_functional=var(f0_sample(:,2));
        median_functional=median(f0_sample(:,2));
        first_quartile_functional=quantile(f0_sample(:,2),0.25);
        third_quartile_functional=quantile(f0_sample(:,2),0.75);

        f0_database=[f0_database;mean_functional,max_functional,range_functional,variance_functional,median_functional,first_quartile_functional,third_quartile_functional,mean_absolute_local_derivative_functional];

        en_file_name=strcat(f0_file_name(1:end-2),"en");
        en_sample=readtable(en_file_name,'FileType','text','ReadRowNames',0,'Delimiter','space');
        en_sample=en_sample{:,:};

        local_derivative_functional=(en_sample(2:end,2)-en_sample(1:end-1,2))./(en_sample(2:end,1)-en_sample(1:end-1,1));

        en_diff_from_zero_mask=(f0_sample(:,2)~=0);
        en_sample=en_sample(f0_diff_from_zero_mask,:);

        mean_absolute_local_derivative=mean(abs(local_derivative_functional(f0_diff_from_zero_mask(1:end-1))));

        mean_functional=mean(en_sample(:,2));
        max_functional=max(en_sample(:,2));
        range_functional=max(en_sample(:,2))-min(en_sample(:,2));
        variance_functional=var(en_sample(:,2));
        median_functional=median(en_sample(:,2));
        first_quartile_functional=quantile(en_sample(:,2),0.25);
        third_quartile_functional=quantile(en_sample(:,2),0.75);
        en_database=[en_database;mean_functional,max_functional,range_functional,variance_functional,median_functional,first_quartile_functional,third_quartile_functional,mean_absolute_local_derivative_functional];
    end
end

database=[f0_database,en_database]; % X of the training database
% Y -class database_target

nb_neighbors = 1; % modify it and test the results

train_test_ratio = 0.6;

[m,n] = size(database) ;
idx = randperm(m) ;
training_database = database(idx(1:round(train_test_ratio*m)),:) ; 
testing_database = database(idx(round(train_test_ratio*m)+1:end),:) ;
training_database_target=database_target(idx(1:round(train_test_ratio*m)),:);
testing_database_target=database_target(idx(round(train_test_ratio*m)+1:end),:);

%%%%%%%%%%%%%%%%%%%% Intra %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model=fitcknn(training_database,training_database_target,'NumNeighbors',nb_neighbors);

yfit = model.predict(testing_database);

% Compute confusion matrix table (2 by 2)
confusion_matrix=confusionmat(testing_database_target,yfit);
disp("Confusion matrice of Baby-Ears");
disp(confusion_matrix);

% Accuracy (number of yfit==testing_database_targets / nb of samples (size
% of testing_database_targets)
accuracy=sum(diag(confusion_matrix))/sum(sum(confusion_matrix));
%accuracy_Baby_ears = ['Accuracy of Baby_ears = ',accuracy];
disp("Accuracy of Baby-ears");
disp(accuracy);

%%%%%%%%%%%%%%%%%%%%%%%%%% CrossCorpus %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% Cross of Kismet with Baby_ears %%%%%%%%%%%%%%%%

modelcc1=fitcknn(training_database_k,training_database_target_k,'NumNeighbors',nb_neighbors);
yfitcc1 = modelcc1.predict(testing_database);

confusion_matrixcc1=confusionmat(testing_database_target,yfitcc1);
disp("Confusion matrice of cross Kismet/Baby-Ears");
disp(confusion_matrixcc1);

accuracycc1=sum(diag(confusion_matrixcc1))/sum(sum(confusion_matrixcc1));
disp("Accuracy Cross Kismet/Baby-Ears");
disp(accuracycc1);


%%%%%%%%%%%%%%%%%% Cross of Baby_ears with Kismet %%%%%%%%%%%%%%%%


modelcc2=fitcknn(training_database,training_database_target,'NumNeighbors',nb_neighbors);
yfitcc2 = modelcc2.predict(testing_database_k);

confusion_matrixcc2=confusionmat(testing_database_target_k,yfitcc2);
disp("Confusion matrice of cross Baby-Ears/Kismet");
disp(confusion_matrixcc2);

accuracycc2=sum(diag(confusion_matrixcc2))/sum(sum(confusion_matrixcc2));
disp("Accuracy Cross Baby-Ears/Kismet");
disp(accuracycc2);


%%%%%%%%%%%%%%%%%% Pooling of Kismet %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

databasepool1 = [training_database_k; training_database];
databasepool2 = [training_database_target_k; training_database_target];

modelcc3 = fitcknn(databasepool1,databasepool2,'NumNeighbors', nb_neighbors);

yfitcc3 = modelcc3.predict(testing_database_k);

conf_matcc3 = confusionmat(testing_database_target_k,yfitcc3);
disp("Cross-Corpus Pool/Kismet test confusion matrice");
disp(conf_matcc3)

accuracycc3=sum(diag(conf_matcc3))/sum(sum(conf_matcc3));
disp("Pooling/Kismet test accuracy")
disp(accuracycc3)

%%%%%%%%%%%%%%%%%%% Pooling of Baby-Ears %%%%%%%%%%%%%%%%%%%%%%%%%%%


yfitcc4 = modelcc3.predict(testing_database);

conf_matcc4 = confusionmat(testing_database_target,yfitcc4);
disp("Cross-Corpus Pool/Baby-ears test confusion matrice");
disp(conf_matcc4);

accuracycc4=sum(diag(conf_matcc4))/sum(sum(conf_matcc4));
disp("Pooling/Baby-ears test accuracy");
disp(accuracycc4);














