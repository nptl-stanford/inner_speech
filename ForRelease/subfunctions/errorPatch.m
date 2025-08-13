function fHandle = errorPatch( xAxis, errData, color, alpha )
    %xAxis is an N x 1 vector
    
    %errData is an N x 2 error region. errData(:,1) is the lower bound,
    %errData(:,2) is the upper bound
    
    %color is a 3 x 1 color
    
    %alpha is a scalar, 0<=alpha<=1
    
    if nargin<4
        alpha = 0.2;
    end
    notNanRows = ~isnan(xAxis) & all(~isnan(errData),2) & all(~isinf(errData),2);
    fHandle = fill([xAxis(notNanRows); flipud(xAxis(notNanRows))]', [errData(notNanRows,1); flipud(errData(notNanRows,2))]', color, ...
        'FaceAlpha', alpha, 'EdgeColor', 'none');
end

