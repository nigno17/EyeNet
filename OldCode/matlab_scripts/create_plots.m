%% network error
close
clear

abs_train_tr3_te1 = importdata('../abs_list.txt');
abs_val_tr3_te1 = importdata('../abs_list_val.txt');

%abs_train_tr3 = importdata('abs_list_tr3.txt');
%abs_val_tr3 = importdata('abs_list_val_tr3.txt');

size(abs_train_tr3_te1)

line_width = 3;
epochs = 10000;

figure('units','normalized','outerposition',[0 0 1 1])
plot(abs_train_tr3_te1(1:epochs), 'b--', 'LineWidth', line_width)
hold on
plot(abs_val_tr3_te1(1:epochs), 'r', 'LineWidth', line_width)

%plot(abs_train_tr3(1:epochs), 'r--', 'LineWidth', line_width)
%plot(abs_val_tr3(1:epochs), 'r', 'LineWidth', line_width)

title('Euclidian distance', 'FontSize', 25)
xlabel('Epochs', 'FontSize', 20)
ylabel('Distance (m)', 'FontSize', 20)
legend({'Train','Test'}, 'FontSize',20)
axis([0 epochs -1000 1000])
set(gca, 'FontSize', 20)
grid on

%% prediction results single axis 

predictions = importdata('../predictions.txt');
ground_truth = importdata('../ground_truth.txt');


figure()
title('Scatter 3D predictions', 'FontSize', 25)
scatter3(predictions(:, 1), predictions(:, 2), predictions(:, 3))

figure()
title('Scatter 3D ground_truth', 'FontSize', 25)
scatter3(ground_truth(:, 1), ground_truth(:, 2), ground_truth(:, 3))

pred_len = length(predictions);

windowSize = 50; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

%predictionsFilteredx = filter(b,a,predictions(:, 1));
predictionsFilteredx = predictions(:, 1);

figure('units','normalized','outerposition',[0 0 1 1])
plot(predictionsFilteredx, 'b--', 'LineWidth', line_width)
hold on
plot(ground_truth(:, 1), 'r', 'LineWidth', line_width)

title('X axis', 'FontSize', 25)
xlabel('Dataset elements', 'FontSize', 20)
ylabel('Position (m)', 'FontSize', 20)
legend({'Predictions','Ground truth'}, 'FontSize',20)
%axis([0 pred_len -0.7 0.6])
set(gca, 'FontSize', 20)
grid on

windowSize = 50; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

%predictionsFilteredy = filter(b,a,predictions(:, 2));
predictionsFilteredy = predictions(:, 2);

figure('units','normalized','outerposition',[0 0 1 1])
plot(predictionsFilteredy, 'b--', 'LineWidth', line_width)
hold on
plot(ground_truth(:, 2), 'r', 'LineWidth', line_width)

title('Y axis', 'FontSize', 25)
xlabel('Dataset elements', 'FontSize', 20)
ylabel('Position (m)', 'FontSize', 20)
legend({'Predictions','Ground truth'}, 'FontSize',20)
%axis([0 pred_len -0.2 0.5])
set(gca, 'FontSize', 20)
grid on

%predictionsFilteredz = filter(b,a,predictions(:, 3));
predictionsFilteredz = predictions(:, 3);

figure('units','normalized','outerposition',[0 0 1 1])
plot(predictionsFilteredz, 'b--', 'LineWidth', line_width)
hold on
plot(ground_truth(:, 3), 'r', 'LineWidth', line_width)

title('Z axis', 'FontSize', 25)
xlabel('Dataset elements', 'FontSize', 20)
ylabel('Position (m)', 'FontSize', 20)
legend({'Predictions','Ground truth'}, 'FontSize',20)
%axis([0 pred_len 0 3.0])
set(gca, 'FontSize', 20)
grid on

figure()
title('Scatter 3D predictions filtered', 'FontSize', 25)
scatter3(predictionsFilteredx, predictionsFilteredy, predictionsFilteredz)

%% prediction results 3D 

% Set up the movie.
writerObj = VideoWriter('out2.avi'); % Name it.
writerObj.FrameRate = 120; % How many frames per second.
open(writerObj); 

figure('units','normalized','outerposition',[0 0 1 1])

windowSize = 100; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

predictionsFilteredx = filter(b,a,predictions(:, 1));
predictionsFilteredy = filter(b,a,predictions(:, 2));
predictionsFilteredz = filter(b,a,predictions(:, 3));
    
 for i = 1:100:(pred_len - 1)
     time_elapsed = i+1;
     plot3(predictionsFilteredx(1:time_elapsed), predictionsFilteredy(1:time_elapsed), predictionsFilteredz(1:time_elapsed), 'b--', 'LineWidth', line_width)
     hold on
     plot3(ground_truth(1:time_elapsed, 1), ground_truth(1:time_elapsed, 2), ground_truth(1:time_elapsed, 3), 'r', 'LineWidth', line_width)
     
     title('Predictions vs Ground truth', 'FontSize', 25)
     xlabel('x (m)', 'FontSize', 20)
     ylabel('y (m)', 'FontSize', 20)
     zlabel('z (m)', 'FontSize', 20)
     legend({'Predictions','Ground truth'}, 'FontSize',20)
     axis([-1 1 -1 1 -1 1])
     set(gca, 'FontSize', 20)
     grid on
     
     frame = getframe(gcf);
     writeVideo(writerObj, frame);
     
     hold off
     
     %pause(0.001)
 end
 
 close(writerObj);
 
 %% prediction results 3D wo movie

figure('units','normalized','outerposition',[0 0 1 1])

windowSize = 100; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

predictionsFilteredx = filter(b,a,predictions(:, 1));
predictionsFilteredy = filter(b,a,predictions(:, 2));
predictionsFilteredz = filter(b,a,predictions(:, 3));

time_elapsed = i+1;
plot3(predictionsFilteredx, predictionsFilteredy, predictionsFilteredz, 'b--', 'LineWidth', line_width)
hold on
plot3(ground_truth(:, 1), ground_truth(:, 2), ground_truth(:, 3), 'r', 'LineWidth', line_width)

title('Predictions vs Ground truth', 'FontSize', 25)
xlabel('x (m)', 'FontSize', 20)
ylabel('y (m)', 'FontSize', 20)
zlabel('z (m)', 'FontSize', 20)
legend({'Predictions','Ground truth'}, 'FontSize',20)
axis([-1 1 -1 1 -1 1])
set(gca, 'FontSize', 20)
grid on
     
 
 close(writerObj);