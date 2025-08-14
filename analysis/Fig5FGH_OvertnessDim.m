clear all;
sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/';
SessionNames = {'t12.2024.04.11','t15.2024.06.14','t16.2024.07.17','t17.2024.12.09'};

participantNames = {'t12','t15','t16','t17'};
participantNamesCap = {'T12','T15','T16','T17'};

participantArrays = {{'ventral6v'},{'ventral6v55b'},{'ventral6v'},{'dorsal6v_combined'}};
participantChans = {[1:64,129:192],[1:64,257:320,129:192,385:448],[65:128,321:384],[1:128,257:384]};

imaginedTrials_P = {[4,7,10,13,16,19,22],[3,6,9,12,15,18,21],[2,5,8,11,14,17,20],[3,6,9,12,15,18,21]};
attemptedTrials_P = {[3,6,9,12,15,18,21],[2,5,8,11,14,17,20],[3,6,9,12,15,18,21],[2,5,8,11,14,17,20]};


%% unsubstracted
unsub_b = {};
unsub_w = {};
for pIdx = 1:length(participantNames)
    dat = load([sessionPath 'Data/interleavedVerbalBehaviors/' SessionNames{pIdx} '_interleavedVerbalBehaviors_raw.mat']);
    if pIdx == 1 || pIdx == 3 % t12 or t16 had 20ms bins
        smoothWidth = 3;
        analysisWindow = [0,25];
    else
        smoothWidth = 6;
        analysisWindow = [0,50];
    end
    
    imaginedTrials = imaginedTrials_P{pIdx};
    attemptedTrials = attemptedTrials_P{pIdx};
    movementCodes = dat.trialCues;

    goTimes = dat.goTrialEpochs(:,1);

    features = [double(dat.binnedTX) double(dat.spikePow)];

    %mean-subtract by block
    blockNums = unique(dat.blockNum);
    for b=1:length(blockNums)
        loopIdx = find(dat.blockNum==blockNums(b));
        features(loopIdx,:) = features(loopIdx,:) - mean(features(loopIdx,:));
    end

    chans = participantChans{pIdx};
    %features = features(:,chans);
    features = gaussSmooth_fast(features,smoothWidth);
    features = zscore(features);

    features = features(:,participantChans{pIdx});

    movementCodesAll = dat.trialCues;
    goTimesAll = dat.goTrialEpochs(:,1);
    movementNamesAll = dat.cueList;
    
    movementSets = {imaginedTrials_P{pIdx},attemptedTrials_P{pIdx}}; %imagined, attempted
    sortedSetIdx = horzcat(movementSets{:}); 
    useTrl = ismember(movementCodesAll, sortedSetIdx);
    movementCodesSet = movementCodesAll(useTrl);
    [originalIdxs,~,movementCodesSet] = unique(movementCodesSet);
    movementNamesSet = movementNamesAll(originalIdxs);
    [~,~,sortedSetIdx] = unique(sortedSetIdx);
    goTimesSet = goTimesAll(useTrl);
    [ C, L,obj, bino_CI ] = simpleClassify( features, movementCodesSet, goTimesSet, movementNamesSet, ...
                        50, 1, 1, true,sortedSetIdx);

    % aesthetic clean up
    ax.Title.FontWeight = 'normal'
    xticklabels({'ban','choice','day','feel','kite','though','were','ban','choice','day','feel','kite','though','were'})
    yticklabels({'ban','choice','day','feel','kite','though','were','ban','choice','day','feel','kite','though','were'})
    axis equal; 

    colormap jet
    clim([0,1])
    
    % add boxes by group
    boxColors = [173,150,61;
    119,122,205;
    91,169,101;
    197,90,159;
    202,94,74]/255;
    boxColors = [boxColors; 0.8*[0.2667    0.8000    0.5333]; 0.8*[0    0.5333    0.8000]; lines(5)];
    
    currentIdx = 0;
    currentColor = 1;
    for c=1:length(movementSets)
        newIdx = currentIdx + (1:length(movementSets{c}))';
        rectangle('Position',[newIdx(1)-0.5, newIdx(1)-0.5,length(newIdx), length(newIdx)],...
            'LineWidth',5,'EdgeColor',boxColors(currentColor,:));
        currentIdx = currentIdx + length(movementSets{c});
        currentColor = currentColor + 1;
    end
    
    countC = zeros(size(C));
    for i = 1:7
        nReps = length(find(movementCodesAll==imaginedTrials(i)))
        countC(i,:) = C(i,:) * nReps;
    end
    for i = 1:7
        nReps = length(find(movementCodesAll==attemptedTrials(i)));
        countC(i+7,:) = C(i+7,:) * nReps;
    end
    behavCorr = (sum(sum(countC(1:7,1:7))) + sum(sum(countC(8:14,8:14))));
    wordCorr = sum(diag(countC)) + countC(1,8) + countC(8,1) + countC(2,9) + countC(9,2) + countC(3,10) + countC(10,3) + countC(4,11) + countC(11,4) + countC(5,12) + countC(12,5) + countC(6,13) + countC(13,6) + countC(7,14) + countC(14,7);
    [unsub_b_mean, unsub_b_CI] = binofit(behavCorr,sum(sum(countC)));
    [unsub_w_mean, unsub_w_CI] = binofit(wordCorr, sum(sum(countC)));
    unsub_b{pIdx} = [unsub_b_mean, unsub_b_CI];
    unsub_w{pIdx} = [unsub_w_mean, unsub_w_CI];

    % save off conf mat
    %exportPNGFigure(gcf, [sessionPath 'Derived/Figures/Fig5_ConfusionMat_' participantNames{pIdx} '_unsubtracted_all']);
end

%% substracted (projecting off A-B/norm(A-B) to remove exact dimension that maximally separates behaviors)
sub_b = {};
sub_w = {};
P_outs = {};
for pIdx = 1:length(participantNames)
    dat = load([sessionPath 'Data/interleavedVerbalBehaviors/' SessionNames{pIdx} '_interleavedVerbalBehaviors_raw.mat']);
    if pIdx == 1 || pIdx == 3 % t12 or t16 had 20ms bins
        smoothWidth = 3;
        analysisWindow = [0,25];
    else
        smoothWidth = 6;
        analysisWindow = [0,50];
    end
    
    imaginedTrials = imaginedTrials_P{pIdx};
    attemptedTrials = attemptedTrials_P{pIdx};
    movementCodes = dat.trialCues;

    goTimes = dat.goTrialEpochs(:,1);

    features = [double(dat.binnedTX) double(dat.spikePow)];

    %mean-subtract by block
    blockNums = unique(dat.blockNum);
    for b=1:length(blockNums)
        loopIdx = find(dat.blockNum==blockNums(b));
        features(loopIdx,:) = features(loopIdx,:) - mean(features(loopIdx,:));
    end

    chans = participantChans{pIdx};
    %features = features(:,chans);
    features = gaussSmooth_fast(features,smoothWidth);
    features = zscore(features);

    features = features(:,participantChans{pIdx});

    movementCodesAll = dat.trialCues;
    goTimesAll = dat.goTrialEpochs(:,1);
    movementNamesAll = dat.cueList;
    
    
%     % mean of both behaviors
    movementSets = {imaginedTrials_P{pIdx},attemptedTrials_P{pIdx}}; %imagined, attempted
%     %imagined
%     useTrlI = ismember(movementCodesAll, movementSets{1});
%     trlIdxI = find(useTrlI);
%     groupMeanA = mean(allFeatures(trlIdxI,:));
%     %attempted
%     useTrlA = ismember(movementCodesAll, movementSets{2});
%     trlIdxA = find(useTrlA);
%     groupMeanI = mean(allFeatures(trlIdxA,:));
% 
%     overtDimVec = (groupMeanI-groupMeanA)/norm(groupMeanI-groupMeanA);
% 
%     P_o = overtDimVec * overtDimVec';
%     P_out = eye(size(overtDimVec,1)) - P_o;
%     allFeatures_proj = allFeatures * P_out;

    sortedSetIdx = horzcat(movementSets{:}); 
    useTrl = ismember(movementCodesAll, sortedSetIdx);
    movementCodesSet = movementCodesAll(useTrl);
    [originalIdxs,~,movementCodesSet] = unique(movementCodesSet);
    movementNamesSet = movementNamesAll(originalIdxs);
    [~,~,sortedSetIdx] = unique(sortedSetIdx);
    goTimesSet = goTimesAll(useTrl);

    movementSetsSorted = {[2,4,6,8,10,12,14],[1,3,5,7,9,11,13]}
    [ C, L,obj, bino_CI, P_out ] = simpleClassify_removeBehaviorDim( features, movementCodesSet, movementSetsSorted, goTimesSet, movementNamesSet, ...
                        50, 1, 1, true,sortedSetIdx);

    P_outs{pIdx} = P_out;

    % aesthetic clean up
    ax.Title.FontWeight = 'normal'
    xticklabels({'ban','choice','day','feel','kite','though','were','ban','choice','day','feel','kite','though','were'})
    yticklabels({'ban','choice','day','feel','kite','though','were','ban','choice','day','feel','kite','though','were'})
    axis equal; 
    
    colormap jet
    clim([0,1])
    
    % add boxes by group
    boxColors = [173,150,61;
    119,122,205;
    91,169,101;
    197,90,159;
    202,94,74]/255;
    boxColors = [boxColors; 0.8*[0.2667    0.8000    0.5333]; 0.8*[0    0.5333    0.8000]; lines(5)];
    
    currentIdx = 0;
    currentColor = 1;
    for c=1:length(movementSets)
        newIdx = currentIdx + (1:length(movementSets{c}))';
        rectangle('Position',[newIdx(1)-0.5, newIdx(1)-0.5,length(newIdx), length(newIdx)],...
            'LineWidth',5,'EdgeColor',boxColors(currentColor,:));
        currentIdx = currentIdx + length(movementSets{c});
        currentColor = currentColor + 1;
    end

    countC = zeros(size(C));
    for i = 1:7
        nReps = length(find(movementCodesAll==imaginedTrials(i)));
        countC(i,:) = C(i,:) * nReps;
    end
    for i = 1:7
        nReps = length(find(movementCodesAll==attemptedTrials(i)));
        countC(i+7,:) = C(i+7,:) * nReps;
    end
    behavCorr = (sum(sum(countC(1:7,1:7))) + sum(sum(countC(8:14,8:14))));
    wordCorr = sum(diag(countC)) + countC(1,8) + countC(8,1) + countC(2,9) + countC(9,2) + countC(3,10) + countC(10,3) + countC(4,11) + countC(11,4) + countC(5,12) + countC(12,5) + countC(6,13) + countC(13,6) + countC(7,14) + countC(14,7);
    [sub_b_mean, sub_b_CI] = binofit(behavCorr,sum(sum(countC)));
    [sub_w_mean, sub_w_CI] = binofit(wordCorr, sum(sum(countC)));
    sub_b{pIdx} = [sub_b_mean, sub_b_CI];
    sub_w{pIdx} = [sub_w_mean, sub_w_CI];

    % save off conf mat
    %exportPNGFigure(gcf, [sessionPath 'Derived/Figures/Fig5_ConfusionMat_' participantNames{pIdx} '_subtractedSupervised_all']);
end

%% bar plot of word vs behavior accs before and after
%% Word Accuracy change plot

% Create the bar plot
fig = figure('units','inch','position',[0,0,5,5]);
hold on;

% Loop through participants
for pIdx = 1:length(participantNames)
    % Bar positions
    x_unsub = 3 * (pIdx - 1) + 1;
    x_sub = 3 * (pIdx - 1) + 2;

    % Plot bars for unsub and sub conditions
    bar(x_unsub, 100 * unsub_w{pIdx}(1), 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none');
    bar(x_sub, 100 * sub_w{pIdx}(1), 'FaceColor', [0.8, 0.4, 0.6], 'EdgeColor', 'none');

    % Add error bars for unsub condition
    errorbar(x_unsub, 100 * unsub_w{pIdx}(1), ...
             100 * (unsub_w{pIdx}(1) - unsub_w{pIdx}(2)), ...  % Lower error
             100 * (unsub_w{pIdx}(3) - unsub_w{pIdx}(1)), ...  % Upper error
             'k', 'LineStyle', 'none', 'LineWidth', 1.5);

    % Add error bars for sub condition
    errorbar(x_sub, 100 * sub_w{pIdx}(1), ...
             100 * (sub_w{pIdx}(1) - sub_w{pIdx}(2)), ...  % Lower error
             100 * (sub_w{pIdx}(3) - sub_w{pIdx}(1)), ...  % Upper error
             'k', 'LineStyle', 'none', 'LineWidth', 1.5);
     % Add participant label above the bars
    x_center = (x_unsub + x_sub) / 2; % Center above the pair of bars
    y_max = 105; % Maximum height of the pair
    text(x_center, y_max + 5, participantNamesCap{pIdx}, 'HorizontalAlignment', 'center', 'FontSize', 10);
end

% Customize the plot
set(gca, 'XTick', [1,2,4,5,7,8,10,11]); % Set XTick positions for blue bars
set(gca, 'XTickLabel', {'Overtness Dim. Intact', 'Overtness Dim. Removed','Overtness Dim. Intact', 'Overtness Dim. Removed','Overtness Dim. Intact', 'Overtness Dim. Removed','Overtness Dim. Intact', 'Overtness Dim. Removed'}, 'FontSize', 12); % Set labels for each bar
xtickangle(45);
ylabel('Word Accuracy (%)');
box off;

% Improve aesthetics
set(gcf, 'Color', 'w');  % Set figure background to white

% Display the plot
hold off;

%exportPNGFigure(gcf, [sessionPath 'Derived/Figures/Fig5_WordAcc']);
%% Word Accuracy change plot

% Create the bar plot
fig = figure('units','inch','position',[0,0,5,5]);
hold on;

% Loop through participants
for pIdx = 1:length(participantNames)
    % Bar positions
    x_unsub = 3 * (pIdx - 1) + 1;
    x_sub = 3 * (pIdx - 1) + 2;

    % Plot bars for unsub and sub conditions
    bar(x_unsub, 100 * unsub_b{pIdx}(1), 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none');
    bar(x_sub, 100 * sub_b{pIdx}(1), 'FaceColor', [0.8, 0.4, 0.6], 'EdgeColor', 'none');

    % Add error bars for unsub condition
    errorbar(x_unsub, 100 * unsub_b{pIdx}(1), ...
             100 * (unsub_b{pIdx}(1) - unsub_b{pIdx}(2)), ...  % Lower error
             100 * (unsub_b{pIdx}(3) - unsub_b{pIdx}(1)), ...  % Upper error
             'k', 'LineStyle', 'none', 'LineWidth', 1.5);

    % Add error bars for sub condition
    errorbar(x_sub, 100 * sub_b{pIdx}(1), ...
             100 * (sub_b{pIdx}(1) - sub_b{pIdx}(2)), ...  % Lower error
             100 * (sub_b{pIdx}(3) - sub_b{pIdx}(1)), ...  % Upper error
             'k', 'LineStyle', 'none', 'LineWidth', 1.5);

     % Add participant label above the bars
    x_center = (x_unsub + x_sub) / 2; % Center above the pair of bars
    y_max = 105; % Maximum height of the pair
    text(x_center, y_max + 5, participantNamesCap{pIdx}, 'HorizontalAlignment', 'center', 'FontSize', 10);
end

% Customize the plot
set(gca, 'XTick', [1,2,4,5,7,8,10,11]); % Set XTick positions for blue bars
set(gca, 'XTickLabel', {'Motor-intent Dim. Intact', 'Motor-intent Dim. Removed','Motor-intent Dim. Intact', 'Motor-intent Dim. Removed','Motor-intent Dim. Intact', 'Motor-intent Dim. Removed','Motor-intent Dim. Intact', 'Motor-intent Dim. Removed'}, 'FontSize', 12); % Set labels for each bar
xtickangle(45);
ylabel('Behavior Accuracy (%)');
box off;

% Improve aesthetics
set(gcf, 'Color', 'w');  % Set figure background to white

% Display the plot
hold off;

%exportPNGFigure(gcf, [sessionPath 'Derived/Figures/Fig5_BehavAcc']);

