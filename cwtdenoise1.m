function y=cwtdenoise1(x,num,therad)
       [c,l]=wavedec(x,num,'db8'); 
        %C：存放的是近似系数（CA）和细节系数（CD）向量；
        %L：存放的是CA和CD对应的长度；
        %X：需分解的信号
        %num：分解尺度（层数） 
        %取第最后层低频近似系数
        ca_num=appcoef(c,l,'db8',num);    
        %appcoef：提取一维信号的某层近似系数（低频系数）
        %ca5：提取的近似系数
        %C，L：小波分解的结构，见上；
        %N：分解尺度（层数）
        %'db8'：小波名称的字符串。        
        %取各层高频细节系数
        cd_cell=cell(1,num);
        for N=1:num
          cd_cell{1,N}=detcoef(c,l,N);          
        end
        % 阈值获取
        % [thr,sorh,keepapp]=ddencmp('den','wv',y); % 函数ddencmp用于获取信号在消噪或压缩过程中的默认阈值        
        thr=thselect(x,'rigrsure'); % 自适应阈值选择使用Stein的无偏风险估计原理        
        % thr=thselect(y,'heursure'); % 使用启发式阈值选择      
        % thr=thselect(y,'sqtwolog'); % 阈值等于sqrt(2*log(length(X)))       
        % thr=thselect(y,'minimaxi'); % 用极大极小原理选择阈值
%         yhard_cell=cell(1,num);
%           yhard_matrix=[];
            c1=ca_num;
        for N1=num:-1:1
             c1=[c1;wthresh(cd_cell{1,N1},therad,thr)];%拼接矩阵
        end
%         c1=[ca_num;yhard_matrix];%拼接矩阵
        y=waverec(c1,l,'db8');%重构
end