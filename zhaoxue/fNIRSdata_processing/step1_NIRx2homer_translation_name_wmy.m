% Transfer NIRSport raw data to HOMER2 data fromat
% this raw_script was made by pro.lu, revised by wmy. raw_script you can
% find in  http://www.ym.edu.tw/~cflu/CFLu_course.html
clear all
clc;clear;
%% load and tranfer raw data
%使用读取文件夹的形式读取原始数据
datadir=uigetdir('Please select a folder with nirx data'); % please make sure the directory name is same with *.wl1 and *.wl2 files
dirinfo=dir(datadir);
dirinfo(1:2)=[];
for subdat=1:length(dirinfo)
    singdatpath = [datadir filesep dirinfo(subdat).name filesep];  %指定每一个数据位置路径
    ind=find(datadir==filesep);
    filename=datadir(ind(end)+1:end); % 获取每一个数据文件
    cd (singdatpath) %将matlab的路径设置为数据所在路径
    filename1=dir('*.wl1');%获取每个文件夹中具体的以.wl1结尾的数据
    filename2=dir('*.wl2');
    filename3=dir('*.evt');
    wl1=load([datadir filesep  dirinfo(subdat).name  filesep filename1.name ]); % %导入wl1数据
    wl2=load([datadir filesep  dirinfo(subdat).name  filesep filename2.name ]); %导入wl2数据
    event=load([datadir filesep dirinfo(subdat).name filesep filename3.name ]); % 导入事件相关信息数据
    fs=7.81;% 实验的采样率
    Ts=1/fs;%获取周期
    t=[0:Ts:(size(wl1,1)-1)*Ts]'; % 计算获取记录的时间
    d=[wl1 wl2]; %wl1和wl2的数据矩阵

    SD.Lambda=[760 850]; % The Wavelengths in the same order as you arranged your 'd' matrix above. 

    % you might need to figure out how your sensor arragement should be made
    % for the SD.SrcPos and SD.DetPos arrays. 
    % the coordinate can be defined using EIguide coordinates
    %以下坐标是通过获取在线采集时得到的相应坐标（根据朱帅鹏数据来）
     SD.SrcPos=[-5.4278 2.7100 1.0180;
                -8.0165 5.9870 0.7943;
                -6.8752 -3.4216 0.3389;                
                -10.0009 0.4637 1.2263;                       
                7.8601 6.2049 0.1731;
                5.8198 2.6965 0.7093;
                9.9299 0.7148 0.7460;
                6.9102 -3.6248 0.0595];    


    SD.DetPos=[-5.2307 5.4896 0.7457;
               -9.0177 3.3373 1.3413;
               -6.1845 -0.3169 1.0451;
               -10.0943 -2.1691 0.6347;
                4.9603 5.8743 0.0288;
                8.8521 3.3135 0.5756;
                6.2751 -0.3425 0.7750;
                10.0710 -2.4432 0.0868]; % The x y z coordinates of your Detector positions. x y z coordinates of [D1; D2; D3; D4 ;D5; D6; D7;D8;D9]
        SD.nSrcs=8; % No. of Sources
        SD.nDets=8; % No. of Detectors


    % SD.MeasList is a (nSrcs . nDets . no of wavelengths) X 4 Matrix     
    m1=[];
    for w=1:2 % 2 wavelength      
        for i=1:SD.nSrcs
            m1=[m1; i*ones(SD.nSrcs,1)]; % the index of sourceNo. , repeat 2 wavelength
        end
    end
    m2=[];
    for w=1:2 % 2 wavelength
        for i=1:SD.nDets
            m2=[m2; [1:SD.nDets]']; % the index of detectorNo. , repeat 2 wavelength
        end
    end
    m3=ones(SD.nSrcs*SD.nDets*2,1); % all ones by default
    m4=[ones(SD.nSrcs*SD.nDets,1);2.*ones(SD.nSrcs*SD.nDets,1)];  % % label of wavelength

    ml=[m1 m2 m3 m4];
    % mask the signal for only preserving the good channel based on S-D design
    % first column is for source label; second column is for detector label
    GoodChanMask=[1 1;
                  1 2;
                  1 3;
                  2 1;
                  2 2;
                  3 3;
                  3 4; 
                  4 2;
                  4 3;
                  4 4;                 
                  5 5;
                  5 6;
                  6 5;
                  6 6;
                  6 7;
                  7 6;
                  7 7;
                  7 8;
                  8 7;
                  8 8];
    GoodChanRow=[];
    for i=1:size(GoodChanMask,1)
        GoodChanRow=[GoodChanRow (GoodChanMask(i,1)-1)*SD.nSrcs+GoodChanMask(i,2)];
    end
    GoodChanRow=sort(GoodChanRow);
    GoodChanRow=[GoodChanRow GoodChanRow+SD.nSrcs*SD.nDets];

    d=d(:,GoodChanRow);
    ml=ml(GoodChanRow,:);

    SD.MeasList=ml;

    % Event file is converted from binary to decimal
    for i=1:size(event,1)
    evt(i,1) = event(i,1);
        %translating binary-markers to decimals
        evt(i,2) = bin2dec([num2str(event(i,9)) num2str(event(i,8)) num2str(event(i,7))...
                            num2str(event(i,6)) num2str(event(i,5)) num2str(event(i,4)) ...
                            num2str(event(i,3)) num2str(event(i,2))]);
    end

    id=find(evt(:,1)==0);
    evt(id,:)=[];

    % you might have to replace the marker values with your own values here.
    m.cond1=find(evt(:,2)==2)';  % Speech
    m.cond2=find(evt(:,2)==4)';  % Speeh + babble
    m.cond3=find(evt(:,2)==8)';  %  babble only
    m.cond4=find(evt(:,2)==10)';  % music

    tevt.cond1=evt(m.cond1,1);  % Speech
    tevt.cond2=evt(m.cond2,1);    % Speeh + babble
    tevt.cond3=evt(m.cond3,1);  % babble only
    tevt.cond4=evt(m.cond4,1);    %  music
    % the placemrks function develops a marker specific array such that it is 1 at the
    % time points at which the event or marker has occured and 0 otherwise.
    s = zeros(length(t),4); % for 2 conditions
    s(tevt.cond1,1)=1;
    s(tevt.cond2,2)=1;
    s(tevt.cond3,3)=1;
    s(tevt.cond4,4)=1;
    aux=t; % This can be any data of length of your 't' variable. 
 
    % s variable has my event markers for the homer to be used and finally you
    % can save the data i n.nirs format as shown.
    safiname ='D:\lzq_nirs\';
    save([ safiname filesep dirinfo(subdat).name '.nirs' ],'t', 'd', 'SD', 's', 'ml', 'aux');
end

disp('done!')