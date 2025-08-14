function [ C, L, obj, bino_CI, P_out ] = simpleClassify_removeBehaviorDim( features, trlCodes, movementSets, eventIdx, conLabels, binWidth, nDecodeBins, startIdx, plotFig, sortIdx )

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
    
    groupMeans = {};
    % get group averages
    for setIdx = 1:length(movementSets)
        movementSet = movementSets{setIdx};
        useTrl = ismember(trlCodes, movementSet);
        trlIdx = find(useTrl);
        groupMeans{setIdx} = mean(allFeatures(trlIdx,:));
    end
    
    meanDiffVec = squeeze((groupMeans{2} - groupMeans{1})/norm(groupMeans{2}-groupMeans{1}));


    P_o = meanDiffVec' * meanDiffVec;
    P_out = eye(size(meanDiffVec,2)) - P_o;
    allFeatures_proj = allFeatures * P_out;

    codeList = unique(trlCodes);
    if nargin<9
        sortIdx = 1:length(codeList);
    end
    
    obj = fitcdiscr(allFeatures_proj,trlCodes,'DiscrimType','diaglinear','Prior',ones(length(codeList),1));
%     obj = fitcdiscr(allFeatures,trlCodes,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus','MaxTime',300));

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
        set(gca,'XTick',1:length(conLabels(sortIdx)),'XTickLabel',conLabels(sortIdx),'XTickLabelRotation',45);
        set(gca,'YTick',1:length(conLabels(sortIdx)),'YTickLabel',conLabels(sortIdx));
        set(gca,'FontSize',12);
        set(gca,'LineWidth',2);
        colorbar;
        title(['Cross-Validated Decoding Accuracy: ' num2str(100*(1-L),3) '%']);
        axis tight;
    end
end

