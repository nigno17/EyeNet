function [ output_args ] = deep_validation()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% rec_path = '/home/rakovicm/workspace/Gaze_Calibration/Tobii_validation/trial_2/';
% fullName = fullfile([rec_path], 'gaze2.json');

% v = VideoReader('Video2.mp4');
% 
% j=0;
% while hasFrame(v)
%     j=j+1;    
%     video = readFrame(v);
%     if(mod(j,1)==0)
%         imshow(video);
%         j
% 
%         pause;
%     end
% end

% load('video.mat');
% 
% file_tobii = fopen(fullName,'r');
% 
% i=0;
% while ~feof(file_tobii)
%     tline = fgetl(file_tobii);
%     
%     try
%         j_obj = jsondecode(tline);
%         sample = j_obj.gp3;
%         i=i+1;
%         t0(:,i) = j_obj.ts;
%         g3d(:,i) = sample;
%     catch
%     end
% end
% fclose(file_tobii);

% load('tobii.mat');


% t0 = t0 - t0(1);

% figure(1);
% clf;
% hold on;
% plot(g3d');
% plot(g3d(2,200:950));
% plot(g3d(3,200:950));

    load('filtered20.mat');

    gaze_3d(:,1) = predictionsFilteredx;
    gaze_3d(:,2) = predictionsFilteredy;
    gaze_3d(:,3) = predictionsFilteredz;
    
    
     point_range_x = [135 233; 298 474; 528 675; 753 1039; 1123 1462; 1542 2142]; 
%     point_range_x = [2512 2772; 2843 3148; 3228 3535; 3607  3922; 3997 4347; 4408 4742]; 
%     point_range_x = [5200 5482; 5574 5789; 5833 6144; 6228 6477;6570 6848;6900 7177]; 

%     point_range_x = [298 474; 528 675; 753 1039; 1123 1462; 1542 2142; 2234 2380]; 
%     point_range_x = [2843 3148; 3228 3535; 3607  3922; 3997 4347; 4408 4742; 4814 5097]; 
%     point_range_x = [5574 5789; 5833 6144; 6228 6477;6570 6848;6900 7177; 7282 7600]; 
    figure(2);
    figure('unit', 'normalized', 'outerposition',[0 0 1 1]);
    hold on;

    mind = min(min(gaze_3d(point_range_x(1,1):point_range_x(end,2),:)));
    maxd = max(max(gaze_3d(point_range_x(1,1):point_range_x(end,2),:)));
    range = maxd-mind;
    mind = mind - 0.1*range;
    maxd = maxd + 0.1*range;

    lw = 4;
    for i = 1 : size(point_range_x,1)
        plot(point_range_x(i,1):point_range_x(i,2),gaze_3d(point_range_x(i,1):point_range_x(i,2),1),'b','LineWidth',lw);
        plot(point_range_x(i,1):point_range_x(i,2),gaze_3d(point_range_x(i,1):point_range_x(i,2),2),'r','LineWidth',lw);
        plot(point_range_x(i,1):point_range_x(i,2),gaze_3d(point_range_x(i,1):point_range_x(i,2),3),'g','LineWidth',lw);
        line([point_range_x(i, 1) point_range_x(i, 1)],[mind maxd]);
        line([point_range_x(i, 2) point_range_x(i, 2)],[mind maxd]);
    end

    title('3D gaze', 'FontSize', 30)
    xlabel('Samples', 'FontSize', 25)
    ylabel('Marker coordinates (mm)', 'FontSize', 25)
    legend({'x ','y','z'}, 'FontSize',25)
    set(gca, 'FontSize', 25)
    grid on
    

    for k = 1 : 6
        point(k,:) = sum(gaze_3d(point_range_x(k,1):point_range_x(k,2),:),1)./(point_range_x(k,2)-point_range_x(k,1)+1);
    end
    
    hold off;
    v11 = point(2,:)-point(1,:);
    v21 = point(3,:)-point(2,:);
    d11 = round(1000*norm(point(2,:)-point(1,:)));
    d21 = round(1000*norm(point(3,:)-point(2,:)));
    d31 = round(1000*norm(point(3,:)-point(1,:)));
    ang1 = round(atan2d(norm(cross(v11,v21)),dot(v11,v21)));
    ang11 = round(rad2deg((cos(dot(v11,v21)/(norm(v11)*norm(v21))))));

    v12 = point(5,:)-point(4,:);
    v22 = point(6,:)-point(5,:);
    d12 = round(1000*norm(point(5,:)-point(4,:)));
    d22 = round(1000*norm(point(6,:)-point(5,:)));
    d32 = round(1000*norm(point(6,:)-point(4,:)));
    ang2 = round(atan2d(norm(cross(v12,v22)),dot(v12,v22)));
    ang21 = round(rad2deg((cos(dot(v12,v22)/(norm(v12)*norm(v22))))));

    title([' d1:' num2str(d11) ':'  num2str(d12) ';' ' d2:' num2str(d21) ':'  num2str(d22) ';' ' d3:' num2str(d31) ':'  num2str(d32) ';' ' a:' num2str(ang11) ':'  num2str(ang21) ';' ],'FontSize', 30);
    
figure(3);
hold on
k=1;
plot3(gaze_3d(point_range_x(k,1):point_range_x(k,2),1),gaze_3d(point_range_x(k,1):point_range_x(k,2),2),gaze_3d(point_range_x(k,1):point_range_x(k,2),3),'r*')
k=2;
plot3(gaze_3d(point_range_x(k,1):point_range_x(k,2),1),gaze_3d(point_range_x(k,1):point_range_x(k,2),2),gaze_3d(point_range_x(k,1):point_range_x(k,2),3),'g*')
k=3;
plot3(gaze_3d(point_range_x(k,1):point_range_x(k,2),1),gaze_3d(point_range_x(k,1):point_range_x(k,2),2),gaze_3d(point_range_x(k,1):point_range_x(k,2),3),'b*')

end

