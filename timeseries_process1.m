function [x_feature_label,y_feature_label]=timeseries_process1(data_select,select_predict_num,num_feature,num_series)

%    num_features=2*select_predict_num;
   num_train=length(data_select)-num_series;
for i=1:num_train-select_predict_num+1
      timefeaturedata= data_select(i:i+num_series-1,end);
%               net_input(i,:)=timefeaturedata(:)';   
      feature_select=data_select(i+num_series-num_feature:i+num_series-1,1:end-1);
      net_input(i,:)=[feature_select(:)',timefeaturedata(:)'];
end    
% disp(size(data_select,1))
% disp((size(data_select,1)-(num_train-select_predict_num+num_series)))
% for N=1:(size(data_select,1)-(num_train-select_predict_num+num_series))
%     disp(N+(num_train+select_predict_num-num_series)+num_series)
%     timefeaturedata11= data_select(N+(num_train-select_predict_num+num_series):N+(num_train-select_predict_num+num_series)+num_series,end);
%     feature_select11=data_select(N+(num_train-select_predict_num+num_feature-num_series):N+(num_train-select_predict_num+snum_series)+num_series,1:end-1);
%     net_input_Pre_y(N,:)=[feature_select11(:)',timefeaturedata11(:)'];
%     disp(net_input_Pre_y)
% end

% for N=(size(data_select,1)-(num_train-select_predict_num+num_series)):-1:1
%     
%     timefeaturedata11= data_select(end-N+1-num_series+1:end-N+1,end);
%     feature_select11=data_select(end-N+1-num_series+1-num_feature:end-N+1,1:end-1);
%     net_input_Pre_y(N,:)=[feature_select11(:)',timefeaturedata11(:)'];
%     disp(net_input_Pre_y)
% end

for i=1:num_train-select_predict_num+1
      timelabel= data_select(i+num_series:i+num_series+select_predict_num-1,end);
      net_output(i,:)=timelabel(:)';    
end
      net_input2=net_input;
     x_feature_label=net_input2;
     y_feature_label=net_output;
end