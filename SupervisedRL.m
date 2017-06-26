clc,
clear;
%%
%Filename to save the generated dataSet
opt.dataSetName = 'experiment.mat';
%Filename to save the RL agent
opt.filename = 'rlMedium.mat';

runReplayExp = 1;
%%
if(exist(opt.dataSetName, 'file') > 0)
    load(opt.dataSetName);
else
    runReplayExp = 0; 
end

if(runReplayExp)
    rl.act = (0.0:0.5:5.5);
    rl.state = [7,6,5,4,2,6,7,4,5];
    rl.hist = 6;
    
    % layers of Deep Value network
    rl.Layers(1) = imageInputLayer([length(rl.state) rl.hist], 'Name', 'StateOfPatient', 'Normalization', 'none');
    rl.Layers(2) = convolution2dLayer(5, 48, 'Name', 'Conv_1', 'Padding', 2);
    rl.Layers(3) = reluLayer('Name', 'Relu_1');
    rl.Layers(4) = fullyConnectedLayer(288, 'Name', 'FC_1');
    rl.Layers(5) = reluLayer('Name', 'Relu_2');
    rl.Layers(6) = fullyConnectedLayer(length(rl.act), 'Name', 'FC_2');
    rl.Layers(7) = regressionLayer('Name', 'QValue4Action');
    
    rl.trainOpt = trainingOptions('sgdm',...
        'InitialLearnRate', 1e-5,...
        'MaxEpochs',50,...
        'MiniBatchSize',24,...
        'Momentum', 0.0,...
        'L2Regularization', 0,...
        'CheckpointPath','');
    
    rl.gamma = 0.95;
    rl.trainingNumber = 0;
    
    rl.net = [];
    
    progressbar;
    trainingLoss = zeros(length(dataSet), 1);
    for idx = 1:length(dataSet)
        
        batch = randperm(length(dataSet{idx}.r));
        
        if(~isempty(rl.net))
            qsTarget = rl.net.predict(dataSet{idx}.s(:, :, 1, batch));
        else
            qsTarget = randn([length(dataSet{idx}.r), length(rl.act)]);
        end
        
        idxTarget = 0;
        for bb = batch
            idxTarget = idxTarget + 1;
            % for each action the RL took, set the new target
            if(dataSet{idx}.aType(bb) == 1 || isempty(rl.net))
                qsTarget(idxTarget, dataSet{idx}.a(bb)) = dataSet{idx}.r(bb);
            else
                qsTarget(idxTarget, dataSet{idx}.a(bb)) = dataSet{idx}.r(bb) + ...
                    rl.gamma*max(rl.net.predict(dataSet{idx}.sp(:, :, 1, bb)));
            end
        end
        
        if(isempty(rl.net))
            [rl.net, info] = trainNetwork(...
                dataSet{idx}.s(:, :, 1, batch),...
                qsTarget,...
                rl.Layers,...
                rl.trainOpt);
        else
            [rl.net, info] = trainNetwork(...
                dataSet{idx}.s(:, :, 1, batch),...
                qsTarget,...
                rl.net.Layers,...
                rl.trainOpt);
        end
        
        trainingLoss(idx) = sum(info.TrainingLoss)/length(info.TrainingLoss);
        
        rl.trainingNumber  = rl.trainingNumber  + 1;
        
        progressbar(idx/length(dataSet));
    end
    
    save(['rl_' datestr(now,'yymmdd-HHMMSS') '.mat'], 'rl')
    save(opt.filename,'rl')
    
    figure(3)
    plot(1:length(dataSet), trainingLoss)
end