function [ C, L, obj, bino_CI ] = simpleClassify( features, trlCodes, eventIdx, conLabels, binWidth, nDecodeBins, startIdx, plotFig, sortIdx, discrimType, gamma)

    if nargin<8
        plotFig = true;
    end
    
    dataIdxStart = startIdx+(1:(binWidth));
    allFeatures = [];
    for t=1:length(trlCodes)
        tmp = [];
        dataIdx = dataIdxStart;
        for binIdx=1:nDecodeBins
            loopIdx = dataIdx + eventIdx(t);
            tmp = [tmp, mean(features(loopIdx,:),1)];

            if binIdx<nDecodeBins
                dataIdx = dataIdx + binWidth;
            end
        end

        allFeatures = [allFeatures; tmp];
    end

    codeList = unique(trlCodes);
    if nargin<9
        sortIdx = 1:length(codeList);
    end
    
    obj = fitcdiscr(allFeatures,trlCodes,'DiscrimType','diaglinear','Prior',ones(length(codeList),1));
    %obj = fitcdiscr(allFeatures,trlCodes,'DiscrimType','diaglinear','OptimizeHyperparameters' ,'auto','Prior',ones(length(codeList),1),'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
    cvmodel = crossval(obj);
    L = kfoldLoss(cvmodel);
    predLabels = kfoldPredict(cvmodel);

    C = confusionmat(trlCodes, predLabels);
    for rowIdx=1:size(C,1)
        C(rowIdx,:) = C(rowIdx,:)/sum(C(rowIdx,:));
    end
    
    [~,bino_CI]=binofit(sum(predLabels==trlCodes),length(trlCodes));
    C = C(sortIdx, sortIdx);
    
    if plotFig
        figure('Position',[212   524   808   567]);
        hold on;

        imagesc(C);
        set(gca,'XTick',1:length(conLabels),'XTickLabel',conLabels(sortIdx),'XTickLabelRotation',45);
        set(gca,'YTick',1:length(conLabels),'YTickLabel',conLabels(sortIdx));
        set(gca,'FontSize',12);
        set(gca,'LineWidth',2);
        colorbar;
        title(['Cross-Validated Decoding Accuracy: ' num2str(100*(1-L),3) '%']);
        axis tight;
    end
end

