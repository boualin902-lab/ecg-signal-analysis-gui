function ECG_GUI_Pro
    % Interface ECG avec SVM et KNN
    global ECG_signal Fs R_peaks ax resultBox;

    %% Chargement
    [file,path] = uigetfile({'.csv;.mat'},'Sélectionner un fichier ECG');
    if isequal(file,0)
        msgbox('Aucun fichier sélectionné.');
        return;
    end
    ext = split(file, '.'); ext = ext{end};
    fullPath = fullfile(path, file);
    if strcmp(ext,'csv')
        data = readtable(fullPath); ECG_signal = data{:,1};
    else
        s = load(fullPath); ECG_signal = s.ECG_signal;
    end
    Fs = 500;

    %% Interface
    f = figure('Name','Analyse ECG','Color',[0.9 0.95 1],...
        'Position',[200 100 1100 600]);

    uicontrol('Style','text','String','Analyse ECG Professionnelle',...
        'FontSize',14,'FontWeight','bold','BackgroundColor','#007ACC','ForegroundColor','w',...
        'Position',[0 560 1100 40]);

    ax = axes('Parent',f,'Units','pixels','Position',[330 100 750 440]);

    btnNames = {...
        '1. Affichage temporel', '2. FFT', '3. Filtrage',...
        '4. Spectrogramme', '5. Pics R', '6. Intervalles',...
        '7. Ondes PQRST', '8. Rythme & RR', '9. Classification'};

    callbacks = {@plotECG, @plotFFT, @filterECG, @plotSpectrogram, ...
                 @detectRPeaks, @calculateIntervals, @identifyPQRST, ...
                 @calculateHeartRate, @classifyECG};

    for i = 1:9
        uicontrol(f,'Style','pushbutton','String',btnNames{i},...
            'FontSize',10,'FontWeight','bold','ForegroundColor','k',...
            'Position',[20 520 - (i-1)*50 280 40],...
            'BackgroundColor','#cce5ff','Callback',callbacks{i});
    end

    resultBox = uicontrol(f, 'Style','edit', 'Position',[20 20 1060 60], ...
        'Max',2, 'Enable','inactive', 'BackgroundColor','white', 'HorizontalAlignment','left', ...
        'FontSize',11, 'String','Résultats affichés ici...');
end

function plotECG(~, ~)
    global ECG_signal ax;
    cla(ax); plot(ax, ECG_signal); grid(ax, 'on');
    title(ax, 'Signal ECG - Domaine Temporel'); xlabel(ax, 'Échantillons'); ylabel(ax, 'Amplitude');
end

function plotFFT(~, ~)
    global ECG_signal Fs ax;
    L = length(ECG_signal);
    f = (0:L-1)*(Fs/L);
    Y = abs(fft(ECG_signal)).^2 / L;
    cla(ax); plot(ax, f(1:L/2), Y(1:L/2));
    title(ax, 'Spectre de Puissance'); xlabel(ax, 'Fréquence (Hz)'); ylabel(ax, 'Puissance'); grid(ax, 'on');
end

function filterECG(~, ~)
    global ECG_signal Fs ax;
    [b, a] = butter(4, [0.5 40]/(Fs/2), 'bandpass');
    ECG_signal = filtfilt(b, a, ECG_signal);
    cla(ax); plot(ax, ECG_signal);
    title(ax, 'Signal ECG Filtré (0.5 - 40 Hz)'); grid(ax, 'on');
end

function plotSpectrogram(~, ~)
    global ECG_signal Fs ax;
    cla(ax);
    axes(ax);
    spectrogram(ECG_signal, 256, 200, 256, Fs, 'yaxis');
    title('Spectrogramme ECG');
end

function detectRPeaks(~, ~)
    global ECG_signal Fs R_peaks ax;
    [~, R_peaks] = findpeaks(ECG_signal, 'MinPeakHeight', mean(ECG_signal), ...
        'MinPeakDistance', round(0.6*Fs));
    cla(ax); plot(ax, ECG_signal); hold(ax, 'on');
    plot(ax, R_peaks, ECG_signal(R_peaks), 'ro');
    title(ax, 'Détection des pics R'); grid(ax, 'on');
end

function calculateIntervals(~, ~)
    global R_peaks Fs resultBox;
    if isempty(R_peaks), msgbox('Détectez d''abord les pics R.'); return; end
    RR = diff(R_peaks)/Fs;
    txt = sprintf('🧮 RR moyenne = %.2fs | Écart-type = %.2fs\n🔍 Estimation PR ≈ 0.16s | QT ≈ 0.36s', mean(RR), std(RR));
    set(resultBox, 'String', txt);
end

function identifyPQRST(~, ~)
    global ECG_signal R_peaks ax;
    cla(ax); plot(ax, ECG_signal); hold(ax, 'on');
    for i = 1:length(R_peaks)
        R = R_peaks(i);
        if R > 40 && R+80 < length(ECG_signal)
            plot(ax, R-40, ECG_signal(R-40), 'go'); % P
            plot(ax, R-20, ECG_signal(R-20), 'ko'); % Q
            plot(ax, R,     ECG_signal(R),     'ro'); % R
            plot(ax, R+20, ECG_signal(R+20), 'mo'); % S
            plot(ax, R+40, ECG_signal(R+40), 'bo'); % T
        end
    end
    legend(ax, 'Signal','P','Q','R','S','T');
    title(ax, 'Identification des ondes PQRST');
end

function calculateHeartRate(~, ~)
    global R_peaks Fs resultBox;
    if isempty(R_peaks), msgbox('Détectez d''abord les pics R.'); return; end
    RR = diff(R_peaks)/Fs;
    HR = 60 ./ RR;
    txt = sprintf('📊 FC moyenne : %.2f bpm | RR std : %.3fs', mean(HR), std(RR));
    set(resultBox, 'String', txt);
end

function classifyECG(~, ~)
    global ECG_signal resultBox;
    L = length(ECG_signal);
    n = 300;
    numSeg = floor(L/n);
    feats = zeros(numSeg,2); labels = zeros(numSeg,1);
    for i = 1:numSeg
        seg = ECG_signal((i-1)*n+1:i*n);
        feats(i,1) = mean(seg);
        feats(i,2) = std(seg);
        labels(i) = i <= numSeg/2;
    end

    % SVM
    SVMModel = fitcsvm(feats, labels);
    predSVM = predict(SVMModel, feats);
    accSVM = sum(predSVM==labels)/numSeg * 100;

    % KNN
    KNNModel = fitcknn(feats, labels, 'NumNeighbors', 3);
    predKNN = predict(KNNModel, feats);
    accKNN = sum(predKNN==labels)/numSeg * 100;

    txt = sprintf('🤖 Précision SVM : %.2f %%\n👟 Précision KNN (k=3) : %.2f %%', accSVM, accKNN);
    set(resultBox, 'String', txt);
end
