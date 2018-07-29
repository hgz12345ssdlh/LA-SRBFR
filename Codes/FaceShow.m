clc;
clear all;

numTrainee = 40;
path = 'F:\校文件\数学\线性代数\Project\CroppedYale';

Train = zeros(32256, numTrainee);
countTrain = 0;
face_id_str = sprintf('yaleB01');
files = dir(fullfile(path, face_id_str, ...
                     strcat(face_id_str, '_P00A*.pgm')));

choices = randperm(length(files), numTrainee);

for j = 1:length(choices)
    image = double(imread(fullfile(path, face_id_str, ...
                                   files(choices(j)).name)));
    image = image(:);
    countTrain = countTrain + 1;
    Train(:,countTrain) = image;
end

[N1, m1] = size(Train);
ratio = norm(mean(Train, 2));

normTrain = zeros(1, countTrain);
for i = 1:m1
    normTrain(1,i) = norm(Train(:,i));
end
Train = Train ./ repmat(normTrain, N1, 1);

meanTrain = mean(Train, 2);
meanFace = reshape(int8(meanTrain * ratio), 192, 168);
figure(1);
title('meanFace');
imshow(meanFace);

standX = bsxfun(@minus, Train', mean(Train'));
covXT = (standX * standX');
[V, D] = eig(covXT);
COEFF = standX' * fliplr(V);

for i = 1:4
    eigFace(:,:,i) = reshape(int8(COEFF(:,i) * 2000), 192, 168);
    figure(i+1);
    title(sprintf('eigFace %d', i));
    imshow(eigFace(:,:,i));
end