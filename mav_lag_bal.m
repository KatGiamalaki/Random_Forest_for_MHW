for vars_lag = [2, 7, 14, 30, 50, 70, 100, 140, 170, 200, 230, 270, 310, 365, 390, 450, 500, 560, 600, 660, 730] % in days
    
    vars_lag

    %% load & reshape 7 
    load('mat_19_unbalanced.mat', 'mat_19')
    mat_19 = permute(mat_19,[2 1 3]);
   
    %% Calculate moving average on initial dataset
    movav_19(:,:,1:6)=mat_19(:,:,1:6);
    movav_19(:,:,7:12)=movmean(mat_19(:,:,7:12),[7 0]);
    movav_19(:,:,13)=mat_19(:,:,13);
    movav_19=permute(movav_19,[2 1 3]);

    %% Make multi lags

    mhw_lag = 1; 

    mat_lags=movav_19(:,1:size(movav_19,2)-vars_lag,1:12);
    mat_lags(:,size(mat_lags,2)+1:size(movav_19,2),:)= NaN;
    mat_lags(:,1:size(mat_lags,2)-vars_lag,13)=movav_19(:,vars_lag+1:end,13);

    idx=isnan(mat_lags(1,:,12));
    mat_lags(:,idx,:)=[];

    eval(['all_lag_' num2str(vars_lag) '_19=mat_lags;']);
    %eval(['all_lag_' num2str(vars_lag) '_19=reshape(all_lag_' num2str(vars_lag) '_19,[size(movav_19,1), size(movav_19,2)-vars_lag, size(mat_19,2)]);']);


    clearvars ans mat_lags temp m1 t1 st1 idx mhw_lag

    %% balance 
    tic
    eval(['var1=all_lag_' num2str(vars_lag) '_19;']);

    a=1;
    for ii=1:size(var1,1)
        for jj=1:size(var1,2)
             if var1(ii,jj,13)~=0
                 mat1(a,:)=var1(ii,jj,:);
                 idx = find(var1(ii,:,13)==0 & var1(ii,:,1)==var1(ii,jj,1) & var1(ii,:,2)==var1(ii,jj,2));
                 kk = find(idx>jj); 
                 if isempty(kk)
                     continue
                 elseif size(kk,2)==1
                    kk1=kk(1,1);
                    a=a+1;
                    mat1(a,:)=var1(ii,idx(kk1),:);
                    a=a+1;
                 elseif size(kk,2)>1
                      kk1=kk(1,1);
                      kk2=kk(1,2);  

                      a=a+1;
                      mat1(a,:)=var1(ii,idx(kk1),:);
                      a=a+1;
                     mat1(a,:)=var1(ii,idx(kk2),:);
                     a=a+1;
                 end

                 elseif var1(ii,jj,12)==0
                     continue
             end
        end
    end
    % % % Add 2019 % % % 
    all_csv=reshape(var1,[size(var1,1)*size(var1,2) 13]);

    var2=all_csv;
    mat2=mat1;
    kk=2019;
    idx=var2(:,3)==2019 & var2(:,13)~=0; aa=sum(idx);
    mat2=var2(idx,:);

    mattest=mat1;
    mattest(size(mattest,1):size(mattest,1)+size(mat2,1)-1,:)=mat2;

    mat_temp_19=mattest;

    %make less categories
    idx=mat_temp_19(:,13)==4;
    sum(idx)
    mat_temp_19(idx,13)=3;


    %clearvars idx kk mat2 mattest aa 
    % % % 

    save(['movav_' num2str(vars_lag) '_19before.mat'],'mat_temp_19');
    writematrix(mat_temp_19,['movav_' num2str(vars_lag) '_19before.csv']);

    %save(['mat' num2str(vars_lag) '_19new.mat'],'mat_temp_19');
    %writematrix(mat_temp_19,['mat' num2str(vars_lag) '_19new.csv']);


    clearvars -except mat_19
    toc
end
%writematrix(mat_230_19,'mat_230_balanced_19.csv');




