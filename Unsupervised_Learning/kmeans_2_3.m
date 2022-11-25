clear;clc;

X = importdata('ThreeCirclesNew.dat');

[labels, centers] = kmeans(X, 3, 'Start', [X(200, :); ...
    X(500, :); X(800, :)]);

X = [X' centers']';
labels = [labels' 4 5 6]';

figure('visible','off');
gscatter(X(:,1), X(:,2), labels, 'rgbrgb', '...***');
title('K-Means Clustering Scatter-Plot');
legend('Cluster-1', 'Cluster-2', 'Cluster-3', 'Center-1', ...
    'Center-2', 'Center-3');
saveas(gcf, 'scatterplot-2-3.png');
