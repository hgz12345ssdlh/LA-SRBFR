clc;
clear;

path = 'F:\У�ļ�\��ѧ\���Դ���\Project\CroppedYale';
numTrainee = 40;

Accu = zeros(1, 5);
for i = 1:5
%         fprintf('Test %d Starts...\n', i);
    Accu(i) = SRBFR(numTrainee, path);
end
meanAccu = mean(Accu);
