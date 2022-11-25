close all;
clear all;
clc
%% data import and parameter of W matrix
data=importdata('ThreeCirclesNew.dat');
data=data';
[row, col]=size(data);
sigma2=0.5;
e=1;
format long

%% creating W matrix
for i=1:col
    for j=1:col
        x1=data(1,i);
        y1=data(2,i);
        x2=data(1,j);
        y2=data(2,j);
        dist=sqrt((x1-x2)^2+(y1-y2)^2);
        if dist<e
            W(i,j)=exp(-dist^2/(2*sigma2));
        else
            W(i,j)=0;
        end
    end
end

%% creating the Degree (D) matrix and L matrix
degree=sum(W');
D=zeros(col,col);
for i=1:col
    D(i,i)=degree(i);
end

%% eigen decomposition
L=D-W;
[V,E]=eig(L,D);

%% stacking the 2nd and 3rd eigen vector
vector=V(:,2:3);

%% K means
[labels, centers] = kmeans(vector, 3, 'Start', [vector(200, :); ...
    vector(500, :); vector(800, :)]);
%[labels,cx,cy] = myKmeans(vector',[200,500,800]);

%% figure generation
figure(1), plot(1:length(diag(E)), diag(E));
title("Plot of eigenvalues")
xlabel("order of eigenvalues")
ylabel("Eigenvalues")

figure(2), scatter(1:length(vector),vector(:,1)','r','*');
hold on
scatter(1:length(vector),vector(:,2)','b','o');
hold off
legend('Eigen Vector 2','Eigen Vector 3');
title("Plot of eigenvector of the second lowest eigen value")

figure(3),
gscatter(data(1,:),data(2,:),labels);
title('Spectral Clustering');
legend('Cluster 1', 'Cluster 2', 'Cluster 3');