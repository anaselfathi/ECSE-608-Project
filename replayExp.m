%% replay experimnent
% This script re-train an agent on existing data set
% Author: anas.elfathi@mail.mcgill.ca
clear;
%%
%Filename to save the generated dataSet
opt.dataSetName = 'experiment.mat';
%Filename to save the RL agent
opt.filename = 'rl.mat';

runReplayExp = 1;
%%
if(exist(opt.dataSetName, 'file') > 0)
    load(opt.dataSetName);
else
    runReplayExp = 0;
end

if(exist(opt.filename, 'file') > 0)
    load(opt.filename);
    if(exist('srl', 'var') > 0)
        rl = srl;
        rl.net = cell(1);
        rl.net{1} = srl.net;
    end
else
    runReplayExp = 0;
end

if(runReplayExp)
    rl.trainOpt = trainingOptions('sgdm',...
        'InitialLearnRate', 1e-5,...
        'MaxEpochs',50,...
        'MiniBatchSize',72,...
        'Momentum', 0.0,...
        'L2Regularization', 0,...
        'CheckpointPath','');
    
    progressbar;
    trainingLoss = zeros(length(dataSet), 1);
    for idx = 1:length(dataSet)
        
        if(idx > 200 && idx < 1200)
            
            batch = randperm(length(dataSet{idx}.r));
            
            qsTarget = rl.net{1}.predict(dataSet{idx}.s(:, :, 1, batch));
            idxTarget = 0;
            for bb = batch
                idxTarget = idxTarget + 1;
                % for each action the RL took, set the new target
                if(dataSet{idx}.aType(bb) == 1)
                    qsTarget(idxTarget, dataSet{idx}.a(bb)) = dataSet{idx}.r(bb);
                else
                    qsTarget(idxTarget, dataSet{idx}.a(bb)) = dataSet{idx}.r(bb) + ...
                        rl.gamma*max(rl.net{1}.predict(dataSet{idx}.sp(:, :, 1, bb)));
                end
            end
            
            [rl.net{1}, info] = trainNetwork(...
                dataSet{idx}.s(:, :, 1, batch),...
                qsTarget,...
                rl.net{1}.Layers,...
                rl.trainOpt);            
            
            trainingLoss(idx) = sum(info.TrainingLoss)/length(info.TrainingLoss);
        end
        progressbar(idx/length(dataSet));
    end
    
    figure(3)
    plot(1:length(dataSet), trainingLoss)
    
    save(['backup/rl_' datestr(now,'yymmdd-HHMMSS') '.mat'], 'rl')
    save(opt.filename,'rl')
end