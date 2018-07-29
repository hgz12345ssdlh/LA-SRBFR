clc;
clear;

path = 'F:\校文件\数学\线性代数\Project\CroppedYale';
numTrainee = 40;

% Data Reading
numTestee = 15;
Train = zeros(32256, 38 * numTrainee);
Test = zeros(32256, 38 * numTestee);
countTrain = 0;
countTest = 0;
%     fprintf('Start Image Reading...\n');
for i = 1:39
    face_id_str = sprintf('yaleB%02d', i);
    files = dir(fullfile(path, face_id_str, ...
                         strcat(face_id_str, '_P00A*.pgm')));
    if isempty(files)
        continue
    end
    choices = randperm(length(files), numTrainee + numTestee);
    choiceTrain = choices(1:numTrainee);
    choiceTest = choices(numTrainee+1:end);
    % Training Data Acquiring
    for j = 1:length(choiceTrain)
        image = double(imread(fullfile(path, face_id_str, ...
                                       files(choiceTrain(j)).name)));
        image = image(:);
        countTrain = countTrain + 1;
        Train(:,countTrain) = image;
    end
    % Testing Data Acquiring
    for j = 1:length(choiceTest)
        image = double(imread(fullfile(path, face_id_str, ...
                                       files(choiceTest(j)).name)));
        image = image(:);
        countTest = countTest + 1;
        Test(:,countTest) = image;
    end
end
%     fprintf('Reading Finished, #Train: %d, #Test: %d\n', countTrain, countTest);

% Normalize each sample in Train, memorize the mean
[N1, m1] = size(Train);
normTrain = zeros(1, countTrain);
for i = 1:m1
    normTrain(1,i) = norm(Train(:,i));
end
Train = Train ./ repmat(normTrain, N1, 1);
meanTrain = mean(Train, 2);

% PCA
%     fprintf('Start PCA...\n');
[COEFF, SCORE, ~] = PCA(Train');
%     fprintf('PCA with Accuracy 95%% Finished.\n');
pcaTrain = SCORE';

% Normalize each sample in Test, subtract meanTrain and project onto COEFF
Test = Test(:,any(Test));
[N2, m2] = size(Test);
normTest = zeros(1, countTest);
for i = 1:m2
    normTest(1,i) = norm(Test(:,i));
end
Test = Test ./ repmat(normTest, N2, 1);
Test = bsxfun(@minus, Test, meanTrain);
pcaTest = (Test' * COEFF)';

% Calculate sparse x by feature_sign, and test results
succTest = 0;
memX = zeros(38 * numTrainee, 38 * numTestee);
%     fprintf('Start Testing Accuracy...\n');
for i = 1:countTest
    x = feature_sign(pcaTrain, pcaTest(:,i), ...
                     0.008, zeros(countTrain,1));
    memX(:,i) = x;
    norm_x = zeros(1,38);
    for j = 1:38
        norm_x(j) = norm(x(numTrainee*(j-1)+1:numTrainee*j));
    end
    [~, deterFace] = max(norm_x);
    if deterFace == ceil(i / numTestee)
        succTest = succTest + 1;
    end
end
Accuracy = succTest / countTest;
%  