function exportPNGFigure(fHandle, fileName)
    %exports a figure as .fig and a nice .png
    set(fHandle,'Renderer','painters');
    set(fHandle,'PaperPositionMode','auto','InvertHardcopy','off','Color','w');
    print(fHandle,'-dpng','-r300',[fileName '.png']);
    saveas(fHandle,[fileName '.fig'],'fig');
    saveas(fHandle,[fileName '.svg'],'svg');
end
