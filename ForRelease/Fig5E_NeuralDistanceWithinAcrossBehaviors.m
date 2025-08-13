clear all;
sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/';
SessionNames = {'t12.2024.04.11','t15.2024.06.14','t16.2024.07.17','t17.2024.12.09'};

participantNames = {'t12','t15','t16','t17'};

participantArrays = {{'ventral6v'},{'ventral6v55b'},{'ventral6v'},{'dorsal6v_combined'}};
participantChans = {[1:64,129:192],[1:64,257:320,129:192,385:448],[65:128,321:384],[1:128,257:384]};

imaginedTrials_P = {[4,7,10,13,16,19,22],[3,6,9,12,15,18,21],[2,5,8,11,14,17,20],[3,6,9,12,15,18,21]};
attemptedTrials_P = {[3,6,9,12,15,18,21],[2,5,8,11,14,17,20],[3,6,9,12,15,18,21],[2,5,8,11,14,17,20]};


%%
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
    features = features(:,chans);
    features = gaussSmooth_fast(features,smoothWidth);
    features = zscore(features);

    %features = features * P_outs{pIdx};

    % within behavior 
    for behavior = 1:2
        if behavior == 1
            Trials = imaginedTrials;
        else
            Trials = attemptedTrials;
        end
        idx = 0;
        for i1 = 1:7
            % get stack of trials 
            triali1s = find(dat.trialCues == Trials(i1));
            nTrials = length(triali1s);
            nDims = length(chans);
            trialVecs1 = zeros(nTrials,nDims);
            for trial = 1:nTrials
                trialStart = dat.goTrialEpochs(triali1s(trial),1) + analysisWindow(1);
                trialEnd = dat.goTrialEpochs(triali1s(trial),1) + analysisWindow(2);
                trialFeats = features(trialStart:trialEnd,:);
                trialVecs1(trial,:) = mean(trialFeats,1);
            end
            for i2 = 1:7
                if i2 <= i1
                    continue;
                end
                idx = idx + 1;
                triali2s = find(dat.trialCues == Trials(i2));
                nTrials = length(triali2s);
                trialVecs2 = zeros(nTrials,nDims);
                for trial = 1:nTrials
                    trialStart = dat.goTrialEpochs(triali2s(trial),1) + analysisWindow(1);
                    trialEnd = dat.goTrialEpochs(triali2s(trial),1) + analysisWindow(2);
                    trialFeats = features(trialStart:trialEnd,:);
                    trialVecs2(trial,:) = mean(trialFeats,1);
                end
                [ euclideanDistance, squaredDistance, CI, CIDistribution ] = cvDistance(trialVecs1,trialVecs2,false,'jackknife');
                allDistsWithin(pIdx,idx,behavior) = euclideanDistance;
                allDistsWithinCIs(pIdx,idx,behavior,:) = CI(:,1);
            end
        end
    end
    % across behavior
    dists = zeros(7,1);
    for i1 = 1:7
        aTrial = attemptedTrials(i1);
        % get stack of trials 
        trialas = find(dat.trialCues == aTrial);
        nTrials = length(trialas);
        nDims = length(chans);
        trialVecs1 = zeros(nTrials,nDims);
        for trial = 1:nTrials
            trialStart = dat.goTrialEpochs(trialas(trial),1) + analysisWindow(1);
            trialEnd = dat.goTrialEpochs(trialas(trial),1) + analysisWindow(2);
            trialFeats = features(trialStart:trialEnd,:);
            trialVecs1(trial,:) = mean(trialFeats,1);
        end
        iTrial = imaginedTrials(i1);
        trialis = find(dat.trialCues == iTrial);
        nTrials = length(trialis);
        nDims = length(chans);
        trialVecs2 = zeros(nTrials,nDims);
        for trial = 1:nTrials
            trialStart = dat.goTrialEpochs(trialis(trial),1) + analysisWindow(1);
            trialEnd = dat.goTrialEpochs(trialis(trial),1) + analysisWindow(2);
            trialFeats = features(trialStart:trialEnd,:);
            trialVecs2(trial,:) = mean(trialFeats,1);
        end
        [ euclideanDistance, squaredDistance, CI, CIDistribution ] = cvDistance(trialVecs1,trialVecs2,false,'jackknife');
        allDistsAcross(pIdx,i1) = euclideanDistance;
        allDistsAcrossCIs(pIdx,i1,:) = CI(:,1);
    end
end

%%
% Initialize index and create a figure
idx = 1;
figure('Units', 'centimeters', 'Position', [0, 0, 20, 10]);
hold on;

% Define colors for the bars
colorsWithinCovert = [0.85, 0.25, 0.10]; % RGB color for within group bars (reddish color)
colorsWithinOvert = [0.85, 0.5, 0.10]; % RGB color for within group bars (reddish color)
colorsAcross = [0.85, 0.75, 0.10]; % RGB color for across group bars (blue color)

% Plotting for 'Within Group' data
for bIdx = 1:2
    if bIdx == 1
        for pIdx = 1:4
            [MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(allDistsWithin(pIdx,:,bIdx));
            bar(idx, MUHAT, 'FaceColor', colorsWithinCovert, 'EdgeColor', 'none'); % Redefine bar with color
            errorbar(idx, MUHAT, MUCI(2) - MUHAT, MUHAT - MUCI(1), 'k', 'LineWidth', 1.5); % Error bars in black
            idx = idx + 1;
        end
    else
        for pIdx = 1:4
            [MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(allDistsWithin(pIdx,:,bIdx));
            bar(idx, MUHAT, 'FaceColor', colorsWithinOvert, 'EdgeColor', 'none'); % Redefine bar with color
            errorbar(idx, MUHAT, MUCI(2) - MUHAT, MUHAT - MUCI(1), 'k', 'LineWidth', 1.5); % Error bars in black
            idx = idx + 1;
        end
    end
    idx = idx + 1; % Extra space between groups
end

% Plotting for 'Across Group' data
for pIdx = 1:4
    [MUHAT, SIGMAHAT, MUCI, SIGMACI] = normfit(allDistsAcross(pIdx,:));
    bar(idx, MUHAT, 'FaceColor', colorsAcross, 'EdgeColor', 'none'); % Define bar with another color
    errorbar(idx, MUHAT, MUCI(2) - MUHAT, MUHAT - MUCI(1), 'k', 'LineWidth', 1.5); % Error bars in black
    idx = idx + 1;
end

% Set the x-ticks and labels
xticks(1:idx - 1);
xticklabels({'t12','t15','t16','t17',' ','t12','t15','t16','t17',' ','t12','t15','t16','t17'});
ylabel('Cross-Validated Euclidean Distance');
xlabel('Conditions');

% Add a title
title('Cross-Validated Euclidean Distance by Condition');

% Create custom plot objects for the legend to match bar colors
h1 = bar(nan, 'FaceColor', colorsWithinCovert, 'EdgeColor', 'none'); % Dummy bar for "Within Group"
h2 = bar(nan, 'FaceColor', colorsWithinOvert, 'EdgeColor', 'none'); % Dummy bar for "Within Group"
h3 = bar(nan, 'FaceColor', colorsAcross, 'EdgeColor', 'none'); % Dummy bar for "Across Group"

% Add the legend with correct colors
legend([h1, h2,h3], {'Within Inner Speech','Within Attempted Speech', 'Across Group'}, 'Location', 'Best');

% Improve overall aesthetics
set(gca, 'FontSize', 12, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.2);

hold off;
%%

exportPNGFigure(gcf, [sessionPath 'Derived/Figures/Fig5D_allPs_Distances']);
