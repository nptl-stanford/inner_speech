% PSTH plot for speech behaviors
% T12 ch 11 (i6v), T15 ch 23 (55b), T16 ch 84 (i6v), T17 ch (s6v-v) 

sessionPath = '/Users/erinkunz/Desktop/InnerSpeech/Data/isolatedVerbalBehaviors/';
SessionNames = {'t12.2023.08.15','t15.2024.04.07','t16.2024.03.04','t17.2024.12.09'};

participantNames = {'t12','t15','t16','t17'};

behaviorNames = {'attempted','mouthed','imagined1stMotor','imagined1stAuditory','imaginedListening','listening','reading'};
behaviorTitles = {'Attempted','Mimed','Motoric Inner Speech','Auditory Inner Speech','Imagined Listening','Listening','Silent Reading'};

t12behaviorTimes = {[2,3],[2,3],[2,3],[2,3],[2,3],[1.5,1.5],[2,1.5]};
t15behaviorTimes = {[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,0.5],[1.5,0.5]};
t16behaviorTimes = {[2,4.5],[2,3.5],[2,3],[2,3],[2,3],[1.5,1],[2,1]};
t17behaviorTimes = {[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,3],[1.5,0.5],[1.5,0.5]};

behaviorTimes_S = {t12behaviorTimes, t15behaviorTimes, t16behaviorTimes, t17behaviorTimes};

ylims = {[-50,150],[-25,70],[-40, 130],[-40,80]};
ylimsSBP = {[-1,5],[-1.5,4],[-1.5,5],[-1,3]};

chanIdxP = {53, 23, 84, 31};

movementSet = [2:8];

colors = jet(length(movementSet))*0.8;
figure('Units','centimeters','Position',[0    0   18    18*0.3]);


f = 1;
TXCrossing = false;

for participantIdx = 1:4
    behaviorTimes = behaviorTimes_S{participantIdx};
    if participantIdx < 2
        gaussSmoothWidth = 3;
        binWidth = 20;
        binWidthS = 0.02;
        ToHz = 50;
    else
        gaussSmoothWidth = 6;
        binWidth = 10;
        binWidthS = 0.01;
        ToHz = 100;
    end
    chanIdx = chanIdxP{participantIdx};

    for behaviorIdx = 1:7
        % behaviorTime 
        behaviorTime = behaviorTimes{behaviorIdx};
        % load data and get chanIdx to plot
        dat = load([sessionPath SessionNames{participantIdx} '_' behaviorNames{behaviorIdx} '_raw.mat']);
        
        if TXCrossing
            features = double(dat.binnedTX(:,chanIdx));
        else
            features = double(dat.spikePow(:,chanIdx));
        end

        %mean-subtract within block
        blockNums = unique(dat.blockNum);
        for b=1:length(blockNums)
            loopIdx = find(dat.blockNum==blockNums(b));
            features(loopIdx,:) = features(loopIdx,:) - mean(features(loopIdx,:));
        end

        smoothFeat = gaussSmooth_fast(features, gaussSmoothWidth)*ToHz;
        if ~TXCrossing
            smoothFeat = zscore(smoothFeat);
        end

        movementCodes = dat.trialCues;
        
        if behaviorIdx < 6
            goTimes = dat.goTrialEpochs(:,1);
            plottingWindow = [-behaviorTime(1)*1000 / binWidth,behaviorTime(2) *1000 / binWidth];
        else
            % adjust listening / reading times!!!
            goTimes = dat.goTrialEpochs(:,1) - (behaviorTime(1)*1000) / binWidth; 
            plottingWindow = [-500/binWidth, behaviorTime(1)*1000/binWidth];
        end
        
        

        subplot(4,7,f);
        hold on;
        ax = gca;
        ax.FontSize = 5;
    
        ms = movementSet;
        for m=1:length(ms)
            trlIdx = find(movementCodes==ms(m));
            [ concatDat ] = triggeredAvg( smoothFeat, goTimes(trlIdx), plottingWindow );

            timeAxis = (plottingWindow(1):plottingWindow(2))*binWidthS;
            plot(timeAxis, nanmean(concatDat),'Color',colors(m,:),'LineWidth',0.7);
            
            [MUHAT,SIGMAHAT,MUCI,SIGMACI] = normfit(concatDat);
            fHandle = errorPatch( timeAxis', MUCI', colors(m,:), 0.2 );
        end
        if TXCrossing
            ylim(ylims{participantIdx});
        else
            ylim(ylimsSBP{participantIdx});
        end
        %plot([0,0],get(gca,'YLim'),'--k','LineWidth',0.7);
        dashline([0,0],get(gca,'YLim'),1,1,1,1,'color','k')


        xlim([plottingWindow(1)*binWidthS plottingWindow(2)*binWidthS]);

        if participantIdx == 4
            xlabel('Time (s)','FontSize',5);
        end
        if participantIdx == 1
            title(behaviorTitles{behaviorIdx},'FontWeight','Normal','FontSize',5);
        end
        if behaviorIdx == 1
            if TXCrossing
                ylabel('Rate (ΔHz)','FontSize',5)
            else
                %ylabel('Power Density (µV²/Hz)','FontSize',5)
                ylabel('SD (µV²/Hz)','FontSize',5)
            end
        end
        f = f+1;
    end
end

if TXCrossing
    exportPNGFigure(gcf,['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1D_PSTHs']);
else
    exportPNGFigure(gcf,['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1D_PSTHs_SBP']);

end

%%
movementSetNames = {'ban','choice','day','feel','kite','though','were'}
%PSTH legend
figure('Units','Centimeters','Position',[0.3171    0.3833    0.1525*10    0.5094*10]);
hold on;
for m=1:length(movementSetNames)
    plot([0,1],[0,1],'-','Color',colors(m,:),'LineWidth',1);
end
xlim([2 3]); ylim([2 3]);
axis off;
legend(movementSetNames);
exportPNGFigure(gcf,['/Users/erinkunz/Desktop/InnerSpeech/Derived/Figures/Fig1D_PSTHs_legend']);


