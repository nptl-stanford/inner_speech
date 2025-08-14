%% Figure 1 plots
SessionNames = {'t12.2023.08.15','t15.2024.04.07','t16.2024.03.04','t17.2024.12.09'};

% channels by array 
T12chanSets = {[1:64,129:192] , [65:128,193:256]};
T15chanSets = {[1:64,257:320] , [65:128,321:384],[129:192, 385:448], [193:256, 449:512]};
T16chanSets = {[1:64,257:320] , [65:128,321:384],[129:256,385:512]};
T17chanSets = {[1:64,257:320] , [65:128,320:384],[129:256,384:512],[1:128,257:384]};
chanSetss = {T12chanSets, T15chanSets, T16chanSets,T17chanSets};

% array names 
T12chanSetNames = {'ventral6v','dorsal6v'};
T15chanSetNames = {'ventral6v','M1', '55b','dorsal6v'};
T16chanSetNames = {'PEF','ventral6v','6d'};
T17chanSetNames = {'dorsal6v-v','dorsal6v-d','55b','dorsal6v_combined'};
chanSetNamess = {T12chanSetNames, T15chanSetNames, T16chanSetNames,T17chanSetNames};

participantNames = {'t12','t15','t16','t17'};

behaviorNames = {'attempted','mouthed','imagined1stAuditory','imagined1stMotor','imaginedListening','listening','reading'};

t12behaviorTimes = {[2,3],[2,3],[2,3],[2,3],[2,3],[1.5,1.5],[2,1.5]};
t15behaviorTimes = {[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,0.5],[1.5,0.5]};
t16behaviorTimes = {[2,4],[2,3],[2,3],[2,3],[2,3],[1.5,1],[2,1]};
t17behaviorTimes = {[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,0.5],[1.5,0.5]};

behaviorTimes_S = {t12behaviorTimes, t15behaviorTimes, t16behaviorTimes, t17behaviorTimes};

windowSizeMS = 500;
trainP = 0.90;
numFolds = 10;

%%

for participantIdx = 1:length(participantNames)
    % Set window size and slides by participant 
    if participantIdx == 1
        binSize = 20; %t12
        slideSize = 5; % 100 ms slide
        smoothBins = 5;
    else
        binSize = 10;
        slideSize = 10;
        smoothBins = 10;
    end
    windowSize = windowSizeMS / binSize;
    
    %Set chan info, behaviorTimes and paths by participant
    chanSetNames = chanSetNamess{participantIdx};
    chanSets = chanSetss{participantIdx};
    sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/';
    behaviorTimes = behaviorTimes_S{participantIdx};

    for behaviorIdx = 1:length(behaviorNames)
        % set windowSlides for participant based on behavior 
        if behaviorIdx < 6
            % 500ms before / 1000ms after go time for executed behaviors 
            goTime = behaviorTimes{behaviorIdx}(2) *1000;
            goTime = goTime / binSize;
            startBin = -500 / binSize;
            endBin = goTime + 1000/binSize;
            windowSlides = [startBin:slideSize:endBin];
        else
            % delay window for listening and reading
            delayTime = behaviorTimes{behaviorIdx}(1) * 1000;
            delayBinLength = delayTime / binSize;
            windowSlides = [-delayBinLength:slideSize:0];
        end
        % load data for participant and behavior
        dat = load([sessionPath 'Data/isolatedVerbalBehaviors/' SessionNames{participantIdx} '_' behaviorNames{behaviorIdx} '_raw.mat']);
        
        display(behaviorNames{behaviorIdx})
        display(length(dat.trialCues)/8)
        %msFeat = double(dat.features);
        msFeat = double([dat.binnedTX dat.spikePow]);
        %mean-subtract by block
        blockNums = unique(dat.blockNum);
        for b=1:length(blockNums)
            loopIdx = find(dat.blockNum==blockNums(b));
            msFeat(loopIdx,:) = msFeat(loopIdx,:) - mean(msFeat(loopIdx,:));
        end

        msFeat = gaussSmooth_fast(msFeat, smoothBins); 
        msFeat = zscore(msFeat);

        goTimes = dat.goTrialEpochs(:,1);

        movementCodes = dat.trialCues;
        movementSet = [2:8];
        [~,~,sortedSetIdx] = unique(movementSet);
        movementNames = {'Do Nothing','ban','choice','day','feel','kite','though','were'};
        useTrl_all = ismember(movementCodes, movementSet);
        useTrlIdx = find(useTrl_all);
        
        seed = 42;
        for chanSetIdx = 1:length(chanSetNames)

            outputDir = [sessionPath 'Derived/OptimalWindowDecoding/' SessionNames{participantIdx} filesep behaviorNames{behaviorIdx} filesep chanSetNames{chanSetIdx} filesep 'optimalWindowDecoder' filesep num2str(windowSizeMS) filesep num2str(seed) filesep];
            mkdir(outputDir);

            if participantIdx == 1
                features = msFeat(:,chanSets{chanSetIdx})*50;
            else
                features = msFeat(:,chanSets{chanSetIdx})*100;
            end
            
            correctVec = [];
            C = zeros(7,7);


            n = numel(useTrlIdx);
            foldSize = floor(n / numFolds);
            rng(seed);           % Set the seed for reproducibility (any integer seed), note: results vary slightly by seed
            idx = randperm(n); % Randomize the order of trials
            for foldIdx = 1:numFolds
                % Define test set for the current fold
                testStart = (foldIdx - 1) * foldSize + 1;
                testEnd = min(foldIdx * foldSize, n);
                testSetIdx = useTrlIdx(idx(testStart:testEnd));
                
                % Define train set as all other indices
                trainSetIdx = setdiff(useTrlIdx(idx), testSetIdx);

                useTrlTrain = zeros(n,1);
                useTrlTrain(trainSetIdx) = 1;
                useTrlTrain = logical(useTrlTrain);
                movementCodesTrain = movementCodes(useTrlTrain);
                goTimesTrain = goTimes(useTrlTrain);

                useTrlTest = zeros(n,1);
                useTrlTest(testSetIdx) = 1;
                useTrlTest = logical(useTrlTest);
                movementCodesTest = movementCodes(useTrlTest);
                goTimesTest = goTimes(useTrlTest);

                % find optimal window for train set
                best = 1.0;
                for windowIdx = 1:length(windowSlides)
                    window = [windowSlides(windowIdx), windowSlides(windowIdx)+windowSize];
                    try
                        [C, L, obj, bino_CI] = simpleClassify(features, movementCodesTrain, goTimesTrain+window(1), movementNames, window(2)-window(1), 1, 1, false, sortedSetIdx);
                        if L < best
                            best = L;
                            bestWindow = window;
                            save([outputDir filesep 'decoder_' num2str(foldIdx) '.mat'],'L','bino_CI','C','obj','window');
                        end
                    catch
                        %disp('End of trial length')
                    end
                
                end
                % apply optimal window decoder to test set
                bestDecoder = load([outputDir filesep 'decoder_' num2str(foldIdx) '.mat']);
                obj = bestDecoder.obj;
                window = bestDecoder.window;

                nDecodeBins = 1;
                startIdx = 1;
                binWidth = window(2) - window(1);
                dataIdxStart = startIdx+(1:(binWidth));
                eventIdx = goTimesTest + window(1);

                allFeatures = [];
                for t=1:length(movementCodesTest)
                    tmp = [];
                    dataIdx = dataIdxStart;
                    for binIdx=1:nDecodeBins
                        loopIdx = dataIdx + eventIdx(t);
                        if loopIdx(end) > size(features,1)
                            loopIdx = size(features,1) + [-49:1:0]; 
                        end
                        tmp = [tmp, mean(features(loopIdx,:),1)];
        
                        if binIdx<nDecodeBins
                            dataIdx = dataIdx + binWidth;
                        end
                    end
        
                    allFeatures = [allFeatures; tmp];
                end

                predLabels = obj.predict(allFeatures);
                
                correct = [predLabels == movementCodesTest];
                correctVec = [correctVec; correct];
                %add to Confusion Matrix
                for trlIdx = 1:length(predLabels)
                    C(predLabels(trlIdx)-1,movementCodesTest(trlIdx)-1) = C(predLabels(trlIdx)-1,movementCodesTest(trlIdx)-1) + 1;
                end
                for rowIdx=1:size(C,1)
                    C(rowIdx,:) = C(rowIdx,:)/sum(C(rowIdx,:));
                end
            end   
                % mean acc
                acc = sum(correctVec)/length(correctVec);
                % binoCI calculation
               [~,bino_CI]=binofit(sum(correctVec),length(correctVec));
                % CM
                
                % save off vals 
                save([outputDir '10foldval_acc_CI_CM.mat'],'bino_CI','acc','C');
     end
    end
end
%% make decoding table by all participants all arrays


% load data into table

behaviorNames = {'attempted','mouthed','imagined1stMotor','imagined1stAuditory','imaginedListening','listening','reading'};
behaviorLabelsNames = {'Attempted','Mimed','Motoric Inner Speech','Auditory Inner Speech','Imagined Listening','Listening','Silent Reading'};
arrayNames = {'T16-i6v','T12-i6v','T15-i6v','T15-4','T17-s6v-A','T17-s6v-B','T12-s6v','T15-s6v','T15-55b'};
arrayAreaNames = {'ventral6v','ventral6v','ventral6v','M1','dorsal6v-v','dorsal6v-d','dorsal6v','dorsal6v','55b'};
colorIdx = [1,1,1,2,3,3,3,3,4];   

decodingAccs = zeros(length(arrayNames),7);
significance = zeros(length(arrayNames),7);

sessions = {'t16.2024.03.04','t12.2023.08.15','t15.2024.04.07','t15.2024.04.07','t17.2024.12.09','t17.2024.12.09','t12.2023.08.15','t15.2024.04.07','t15.2024.04.07'};
for arrayIdx = 1:length(arrayNames)
    for behaviorIdx = 1:length(behaviorNames)
        %results = load([arrayPaths{arrayIdx} behaviorNames{behaviorIdx} filesep arrayAreaNames{arrayIdx} filesep 'optimalWindowDecoder' filesep num2str(windowSizeMS) filesep '10foldval_acc_CI_CM.mat'])
         results = load([sessionPath 'Derived/OptimalWindowDecoding/' sessions{arrayIdx} filesep behaviorNames{behaviorIdx} filesep arrayAreaNames{arrayIdx} filesep 'optimalWindowDecoder' filesep num2str(windowSizeMS) filesep num2str(seed) filesep '10foldval_acc_CI_CM.mat'])

        decodingAccs(arrayIdx,behaviorIdx) = results.acc;
        if results.bino_CI(1) < 1/7
            significance(arrayIdx,behaviorIdx) = 1;
        end
    end
end

%% plot and label (Fig 1E)
figure('Units','Centimeters','Position',[0 0 10 5]);
decData = 100*flip(decodingAccs);
sigData = flip(significance);
h = imagesc(decData);
ax = gca;
ax.FontSize = 5;

yticklabels(flip(arrayNames));
yticks([1:length(arrayNames)])
xticklabels(behaviorLabelsNames);
for i = 1:length(arrayNames)
    for j = 1:7
        text(j-0.25,i,num2str(decData(i,j),'%.1f'),'FontSize',5)
        if sigData(i,j) > 0
            text(j+0.3,i,'X','Color','r','FontSize',5)
        end
    end
end
xlabel('Behavior')
ylabel('Participant Array')
ticklabels = get(gca,'YTickLabel');

colors = hsv(4)*0.8;
ax = gca;
for i = 1:numel(ax.YTickLabel)
    ax.YTickLabel{i} = sprintf('\\color[rgb]{%1f,%1f,%1f}%s', colors(colorIdx(10-i),:),ax.YTickLabel{i});
end
colorbar;
caxis([100/7 100]);

cmap = brewermap(256, 'Blues');clos
colormap(cmap)

exportPNGFigure(gcf, ['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1E_decodingTable_' num2str(windowSizeMS) '_2']);
%exportPNGFigure(gcf, ['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1E_decodingTable_' num2str(windowSizeMS)]);
%% collect optimal windows

bestWindows = {};
for participantIdx = 1:4
    % Set window size and slides by participant 
    bestWindows{participantIdx} = {};
    if participantIdx == 1
        binSize = 20; %t12
        slideSize = 5; % 100 ms slide
        smoothBins = 5;
    else
        binSize = 10;
        slideSize = 10;
        smoothBins = 10;
    end
    windowSize = windowSizeMS / binSize;
    
    %Set chan info, behaviorTimes and paths by participant
    chanSetNames = chanSetNamess{participantIdx};
    chanSets = chanSetss{participantIdx};
    %sessionPath = SessionPaths{participantIdx};
    sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/';
    behaviorTimes = behaviorTimes_S{participantIdx};
    for behaviorIdx = 1:length(behaviorNames)
        bestWindows{participantIdx}{behaviorIdx} = {};
        for arrayIdx = 1:length(chanSetNames)
            windows = [];
            for n = 1:numFolds
                load([sessionPath 'Derived/OptimalWindowDecoding/' SessionNames{participantIdx} filesep behaviorNames{behaviorIdx} filesep chanSetNames{arrayIdx} filesep 'optimalWindowDecoder/' num2str(windowSizeMS) filesep 'decoder_' num2str(n) '.mat'])
                windows = [windows; window];
            end
            bestWindows{participantIdx}{behaviorIdx}{arrayIdx} = mode(windows);
        end
    end
end

save([sessionPath 'Derived/OptimalWindowDecoding/optimalWindowsAll.mat'],'bestWindows')

%% plot some decoding confusion matrices (Fig 1F)
sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/';

%T16 listening: 
wordsList = {'ban','choice','day','feel','kite','though','were'};
dec = load([sessionPath '/Derived/OptimalWindowDecoding/' SessionNames{3} filesep 'listening' filesep  'ventral6v' filesep 'optimalWindowDecoder' filesep num2str(windowSizeMS) filesep '10foldval_acc_CI_CM.mat']);
figure('Units','centimeters','Position',[0 0.5 4.5 3])
h = imagesc(100*dec.C)

ax = gca;
ax.FontSize = 5;
ax.YTick = [1:7]
ax.XTick = [1:7]
yticklabels(wordsList)
xticklabels(wordsList)

c = colorbar
c.Box = 0;
caxis([0 100])
title(['T16-i6v Listening Decoding Accuracy: ' num2str(100*(dec.acc),'%.1f') '%'],'FontWeight','Normal')

exportPNGFigure(gcf, ['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1F_T16listening_' num2str(windowSizeMS)]);
%%
sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/';

%T12 motoric inner speech: 
wordsList = {'ban','choice','day','feel','kite','though','were'};
dec = load([sessionPath '/Derived/OptimalWindowDecoding/' SessionNames{1} filesep 'imagined1stMotor' filesep  'ventral6v' filesep 'optimalWindowDecoder' filesep num2str(windowSizeMS) filesep '10foldval_acc_CI_CM.mat']);
figure('Units','centimeters','Position',[0 0.5 4.5 3])
h = imagesc(100*dec.C)

ax = gca;
ax.FontSize = 5;
ax.YTick = [1:7]
ax.XTick = [1:7]
yticklabels(wordsList)
xticklabels(wordsList)

c = colorbar
c.Box = 0;
caxis([0 100])
title(['T12-i6v Motor Inner Speech Decoding Accuracy: ' num2str(100*(dec.acc),'%.1f') '%'],'FontWeight','Normal')

exportPNGFigure(gcf, ['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1F_T12MotoricInner_' num2str(windowSizeMS)]);
