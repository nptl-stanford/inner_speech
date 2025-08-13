function [ concatDat ] = triggeredAvg( dat, trigIdx, winIdx )
    idxVec = winIdx(1):winIdx(2);
    concatDat = nan(length(trigIdx),length(idxVec),size(dat,2));
    for t=1:length(trigIdx)
        loopIdx = idxVec + trigIdx(t);
        keepIdx = loopIdx>=1 & loopIdx<=length(dat);
        concatDat(t,keepIdx,:) = dat(loopIdx(keepIdx),:);
    end
end

