%% Main
% This scrip is used to generate random experiments and train the agent
% You can also becnhmark the agent by selecting the appropriate simMode.
% Author: anas.elfathi@mail.mcgill.ca
% Last changed: April 2017
clear;
%% Simulation Option
%0: train,
%1: test best policy,
%2: test expert policy
%3: test random policy,
%4: test do nothing policy,
opt.simMode = 1;
%Benchmark holders
opt.benchName = {'ScoreTest.mat', 'ScoreExpert.mat', 'ScoreRandom.mat', 'ScoreDoNothing.mat'};
%Filename to save the RL agent
opt.filename = 'rl.mat';
%Filename to save the generated dataSet
opt.dataSetName = 'experiment.mat';
%Algorithm type
opt.algo = 1; %0: DQN, 1:DDQN
%Simulation pauses after each day
opt.stepPerStep = 0;
% nominal test
opt.nominalTest = 1;
%% Init variable
dt = 10;

% minimal model
P.GHypo = 5.0;
P.GHyper = 10.0;
P.UbMin = 0.0;
P.UbMax = 5.0;
P.Gs0 = 5.5;
P.Weight = 45; %kg
P.TDD = 70; % total daily dose (Units)
P.MCR = 40;     % Metabolic Clearance rate of insulin mL / Kg / min
P.Ub = 0.55*P.TDD/24;        % Mean Basal Insulin Rate U/h
P.Ipb = 1e6*P.Ub / (60*P.MCR*P.Weight);   % Mean Basal insulin concentration in plasma mU / L
P.MCHO = 180.156; % molar mass for glucose g/mol
P.ICR = 1/10; % Units / g
P.Bio = 1.0;
P.MCHO = 180.156; % molar mass for glucose g/mol
P.Vg = 0.16; % Distrubition volume of the measurement accessible compartiment L/kg
P.Si = P.Bio*P.MCR/(1e3*P.MCHO*P.ICR*P.Vg); % 1/min per mU / L
P.EGP = P.Si*P.Ipb; % mmol per L per min non-insulin dependent glucose utilisation
P.Tau_i = 30; % min
P.Tau_m = 24; % min
P.Tau_sen = 15; % min

Ac = ...
    [-1/P.Tau_sen, 1/P.Tau_sen, 0, 0       , 0                               , 0         , 0       ;...
    0            , 1e-8       , 1, -P.Si   , 0                               , P.Bio/P.Vg, 0       ;...
    0            , 0          , 1e-8, 0       , 0                               , 0         , 0       ;...
    0            , 0          , 0, -1/P.Tau_i, 1e6/(P.Tau_i^2*P.MCR*P.Weight), 0         , 0       ;...
    0            , 0          , 0, 0       , -1/P.Tau_i                      , 0         , 0       ;...
    0            , 0          , 0, 0       , 0                               , -1/P.Tau_m, 1e3/(P.MCHO*P.Weight*P.Tau_m^2) ;...
    0            , 0          , 0, 0       , 0                               , 0         , -1/P.Tau_m];

Bc = ...
    [0;...
    0;...
    0;...
    0;...
    1/60;...
    0;...
    0];

model.A = expm(Ac*dt);
model.B = ((expm(Ac*dt) - eye(size(Ac)))/(Ac))*Bc;
model.B(:,2) = model.A*[0;0;0;0;1;0;0];
model.B(:,3) = model.A*[0;0;0;0;0;0;1];
model.C = [1 0 0 0 0 0 0];
model.D = 0;
model.X0 = [P.Gs0;...             % sensor glucose mmol / L
    P.Gs0;...              % plasma glucose mmol / L
    P.EGP;...
    P.Ipb;...
    P.Ub*P.Tau_i/60;...
    0;...
    0];

% MHE parameters
mheConf.Q = diag([...
    1e-1, ...            % sensor glucose mmol / L
    1.0, ...            % plasma glucose mmol / L
    5e-3, ...              % EGP
    0.5, ...             % Insulin plasma concentraion mU / L
    0.2, ...              % Insulin mass under the skin U
    10.0, ...             % Meal-glucose rate appearance micro-mol / kg / min
    1e2])^2;              % Glucose mass in the stomach micro-mol / kg
mheConf.R = (1.0)^2;
mheConf.X0 = model.X0;
mheConf.P0 = 10*mheConf.Q;
mheConf.Order = 7;

% MPC parameters
mpcConf.NStep = 4*60/dt;
mpcConf.Gy = 0.1/(P.Gs0^2);
mpcConf.Gu = 1/(P.Ub^2);
mpcConf.Gyf = 2*max(mpcConf.Gy,mpcConf.Gu);
%% Genrate Deep RL options
if(exist(opt.filename, 'file') > 0)
    load(opt.filename);
else
    rl.act = (0.0:0.5:5.5);
    rl.state = [7,6,5,4,2,6,7,4,5];
    rl.hist = 60/dt;
    
    % layers of Deep Value network
    rl.Layers(1) = imageInputLayer([length(rl.state) rl.hist], 'Name', 'StateOfPatient', 'Normalization', 'none');
    rl.Layers(2) = convolution2dLayer(5, 48, 'Name', 'Conv_1', 'Padding', 2);
    rl.Layers(3) = reluLayer('Name', 'Relu_1');
    rl.Layers(4) = fullyConnectedLayer(288, 'Name', 'FC_1');
    rl.Layers(5) = reluLayer('Name', 'Relu_2');
    rl.Layers(6) = fullyConnectedLayer(length(rl.act), 'Name', 'FC_2');
    rl.Layers(7) = regressionLayer('Name', 'QValue4Action');
    
    % Create empty 2 networks
    rl.net{1} = [];
    rl.net{2} = [];
    
    rl.trainingNumber = 0;
    rl.gamma = 0.95;
end

if(opt.simMode == 0)   
    rl.totBatches = 1000; % number of episodes to run 
else
    if(opt.simMode == 1) % test best policy
        rl.passive = 0.0; % 1 - passive learning, 0 - active learning
        rl.explore = 0.0; % 1 - explore, 0 - exploit
    elseif(opt.simMode == 2) % test expert policy
        rl.passive = 1.0; % 1 - passive learning, 0 - active learning
        rl.explore = 0.0; % 1 - explore, 0 - exploit
    elseif(opt.simMode == 3) % test random policy
        rl.passive = 1.0; % 1 - passive learning, 0 - active learning
        rl.explore = 1.0; % 1 - explore, 0 - exploit
    elseif(opt.simMode == 4) % test do nothing policy
        rl.passive = 1.0; % 1 - passive learning, 0 - active learning
        rl.explore = 1.0; % 1 - explore, 0 - exploit
    else
        error('unknown simulation mode !!!');
    end
    
    rl.totBatches = 50;
    rl.batchSize = 24*7; % one week of data
    if(opt.nominalTest)
        rl.totBatches = 1;
        rl.batchSize = 24; % one day of data
    end
end
lNet = 1;
tNet = 2;
    
rl.reward = @glucoseInTargetScore;
    
rl.data = cell(rl.totBatches,1);
rl.batchScore = 0;
%% Main loop

nBatch = 0;
while(nBatch < rl.totBatches)

    nBatch = nBatch + 1;
    if(opt.simMode == 0)
        % update learning parameteres
        rl.batchSize = min(rl.trainingNumber + 1,10)*24; % 10 days of data
        rl.memorySize = round(rl.batchSize/3); % Memory for data to use in training (5 days)
        rl.explore = abs(2*(rl.trainingNumber/10 - round(rl.trainingNumber/10))); % 1 - explore, 0 - exploit
        rl.passive = (0.99)^(rl.trainingNumber); % 1 - passive learning, 0 - active learning
        rl.trainOpt = trainingOptions('sgdm',...
            'InitialLearnRate', max(1e-2*(0.99)^(rl.trainingNumber), 1e-5),...
            'MaxEpochs',5*min(2*rl.memorySize,24),...
            'MiniBatchSize',min(2*rl.memorySize,24),...
            'Momentum', 0.0,...
            'L2Regularization', 0.0,...
            'CheckpointPath','');
        
        rl.net{tNet} = rl.net{lNet};
    end

    progressbar;
    sample = 0;
    while(sample < rl.batchSize)
        T = (rl.batchSize-sample)*60; % minutes
        N = T/dt;
        
        % start of the simulation is random between midnight and 6am
        dayStart = randi(6);
        if(opt.nominalTest)
            dayStart = 5;
        end
        
        % initialize a patient
        VP = VirtualPatient(0); %0: minimal
        VP.setNoise(0.5*rand(1), 0.5*rand(1));  % proess / sensor
        if(opt.nominalTest)
            VP.setNoise(0.0, 0.0);  % proess / sensor
        end
        % initialize Moving Horizon Estimator for state estimation
        MHE = MHE_LTI(model, mheConf);
        MHE.ConstraintInit([],[],-eye(length(model.X0)),zeros(size(model.X0)));
        % initialize Model Predective Controller for state estimation
        model.C = [0 1 0 0 0 0 0];
        MPC = MPC_LQG(model, mpcConf);
        model.C = [1 0 0 0 0 0 0];
        
        % intialize state holder
        sys.Time = zeros(N,1);
        sys.Glucose = zeros(N,1);
        sys.I_basal = zeros(N,1);
        sys.I_bolus = zeros(N,1);
        sys.I_Carbs = zeros(N,1);
        X = zeros(length(VP.getX0()), N);
        X(:,1) = VP.getX0();
        UBolus = zeros(N, 1);
        
        % start simulaion
        for k = 1:N
            sys.Time(k) = k*dt;

             % every beginning of day
            if(mod(sys.Time(k)-dt, 24*60) == 0)
                % shuffle the random function 
                rng('shuffle');
                                
                % Meals for the day are randomely chosen (q in (g) and time in minutes)
                meals.q = [round(40 + (60-40)*rand(1))... % breakfast
                    round(60 + (100-60)*rand(1))...       % lunch
                    round(0 + (25)*rand(1))...            % snack
                    round(10 + (60-10)*rand(1))];         % dinner
                meals.t = ([round(7 + (9-7)*rand(1),1)...
                    round(12 + (13-12)*rand(1),1)...
                    round(16 + (18-16)*rand(1),1)...
                    round(19 + (23-19)*rand(1),1)] ...
                    - dayStart)*60 + sys.Time(k) - dt;
                
                % error percentage on carb counting per day
                meals.error = min(max(0.25+0.2*randn(1), -0.1), 0.5);
                
                if(opt.nominalTest)
                    % Meals for the day are randomely chosen (q in (g) and time in minutes)
                    meals.q = [40 ... % breakfast
                        80 ...       % lunch
                        10 ...            % snack
                        60];         % dinner
                    meals.t = ([7.5 ...
                        12.5 ...
                        17.5 ...
                        20.5] ...
                        - dayStart)*60 + sys.Time(k) - dt;
                    
                    % error percentage on carb counting per day
                    meals.error = 0.25;
                end
                
                % holder for last meal time
                meals.lastMealIdx = -1;
            end
                        
            % Measurement
            sys.Glucose(k) = VP.getGlucose();
            
            % MHE
            if(k > 1)
                [~,~,S] = MHE.Run(sys.Glucose(k),...
                    [sys.I_basal(k-1);...
                    sys.I_bolus(k-1);...
                    (1-meals.error)*sys.I_Carbs(k-1)]);
            end
            
            % generate data for RL agent
            if(mod(k, rl.hist) == 0)
                sample = sample + 1;
                
                % policy: save state in (hist x nbrOfStates) format, like an image
                rl.data{nBatch}.s(:, :, 1, sample) = MHE.X(rl.state,end-rl.hist+1:end);

                % save this sample as sprime for previous sample
                if(k > rl.hist + 1)
                    % save resulting reward
                    rl.data{nBatch}.r(sample - 1) = rl.reward(sys.Glucose(k-rl.hist+1:k));
                    % save resulting
                    rl.data{nBatch}.sp(:, :, 1, sample - 1) =  rl.data{nBatch}.s(:, :, 1, sample);
                end
                
                % if end of simulation break
                if(sys.Glucose(k) < 1 && ~opt.nominalTest)
                    sample = sample - 1; % disregard the last s(:, :, 1, sample)
                    rl.data{nBatch}.aType(sample) = 1; % terminal is the highest priority
                    break;
                else
                    rl.data{nBatch}.aType(sample) = 0; % normal prioirity
                end
            end
            
            % Run the MPC controller for basal action
            sys.I_basal(k) = MPC.Run(MHE.XLast,...
                P.Gs0*ones(MPC.N, 1),...
                P.GHypo,...
                P.UbMin,...
                P.UbMax,...
                P.Ub);
            
            % Run RL Agent policy
            if(mod(k, rl.hist) == 0)
                % explore: Discover new states/actions
                if(rand(1) < rl.explore)
                    % if no meal, choose an action with more chance for it
                    % to be do-nothing action
                    if(meals.lastMealIdx < 0)
                        rl.data{nBatch}.a(sample) = max(min(floor(1 + length(rl.act)*randn(1)/2), length(rl.act)), 1);
                        % there was meal with mistake
                    else
                        rl.data{nBatch}.a(sample) = randi(length(rl.act));
                    end
                    if(opt.simMode == 4)
                        rl.data{nBatch}.a(sample) = 1;
                    end
                    % exploit: try my best policy
                else
                    % learn from expert policy
                    if(rand(1) < rl.passive)
                        % if no meal
                        if(meals.lastMealIdx < 0)
                            rl.data{nBatch}.a(sample) = 1;
                            % there was meal with mistake
                        else
                            expertAct = 0.5*meals.error*sys.I_Carbs(meals.lastMealIdx)*VP.getPatientProp.ICR;
                            [~, plIdx] =  sort(abs(rl.act - expertAct));
                            rl.data{nBatch}.a(sample) = plIdx(1);
                        end
                        % try the RL  Agent greedy policy
                    else
                        % acts greedy according to best policy from the
                        % learning net
                        [~, idx] = max(rl.net{lNet}.predict(rl.data{nBatch}.s(:, :, 1, sample)));
                        rl.data{nBatch}.a(sample) = idx;
                        
                        % save the agent policy (for visualization)
                        UBolus(k) = rl.act(rl.data{nBatch}.a(sample));
                    end
                end
                
                % act on the policy
                sys.I_bolus(k) = sys.I_bolus(k) + rl.act(rl.data{nBatch}.a(sample));
                
                % Flag the non-meal action as low priority for learning
                if(meals.lastMealIdx < 0)
                    rl.data{nBatch}.aType(sample) = -1;
                end
                
                meals.lastMealIdx = -1;
            end
            
            % Consume the meal !
            if(~isempty(meals.q))
                if(sum(abs(meals.t - sys.Time(k)) < dt/2))
                    sys.I_Carbs(k) = meals.q(abs(meals.t - sys.Time(k)) < dt/2);
                    % human bolusing with mistakes
                    sys.I_bolus(k) = sys.I_bolus(k) + ...
                        round((1 - meals.error) * sys.I_Carbs(k) * VP.getPatientProp.ICR,1);
                    
                    meals.lastMealIdx = k;
                end
            end
            
            if(k < N)
                % Virtual Patient
                X(:,k + 1) = VP.model(...
                    X(:,k),...
                    sys.I_basal(k),...
                    sys.I_bolus(k),...
                    sys.I_Carbs(k),...
                    sys.Time(k):0.025:sys.Time(k)+dt);
            end
            
            % train network
            if(opt.simMode == 0 &&...                              % in train mode
                    sample > rl.trainOpt.MiniBatchSize &&...   % we at least have MiniBatchSize worth of sample
                    mod(k, rl.hist) == 0 &&...                 % every sample increase
                    mod(sample, 12) == 0)                      % only train once a while (every 12 samples)
                                
                % Construct the learning set of size < rl.memorySize
                randomSample = randperm(sample-1);             % randomize my samples
                batch = zeros(1, min(rl.memorySize, length(randomSample)));
                batchLastIdx = 1;
                for priorityIdx = [1, 0, -1]
                    if(batchLastIdx <= rl.memorySize)
                        priorityBatch = randomSample(rl.data{nBatch}.aType(randomSample) == priorityIdx);
                        if(~isempty(priorityBatch))
                            priorityBatch = priorityBatch(1:min(length(priorityBatch),rl.memorySize-batchLastIdx+1));
                            batch(batchLastIdx:batchLastIdx+length(priorityBatch)-1) = priorityBatch;
                            batchLastIdx = batchLastIdx + length(priorityBatch);
                        end
                    end
                end
                
                % construct the traget from tNet
                % start with target equals to expected target in lNet
                if(~isempty(rl.net{lNet}))
                    qsTarget = rl.net{lNet}.predict(rl.data{nBatch}.s(:, :, 1, batch));
                else
                    qsTarget = randn([length(rl.data{nBatch}.r), length(rl.act)]);
                end
                idxTarget = 0;
                for bb = batch
                    idxTarget = idxTarget + 1;
                    % for each action the RL took, set the new target
                    if(rl.data{nBatch}.aType(bb) == 1 || isempty(rl.net{tNet}))
                        qsTarget(idxTarget, rl.data{nBatch}.a(bb)) = rl.data{nBatch}.r(bb);
                    else
                        if(opt.algo == 0)
                            qsTarget(idxTarget, rl.data{nBatch}.a(bb)) = rl.data{nBatch}.r(bb) + ...
                                rl.gamma*max(rl.net{tNet}.predict(rl.data{nBatch}.sp(:, :, 1, bb)));
                        else
                            [~, idx] = max(rl.net{lNet}.predict(rl.data{nBatch}.sp(:, :, 1, bb)));
                            qsp = rl.net{tNet}.predict(rl.data{nBatch}.sp(:, :, 1, bb));
                            qsTarget(idxTarget, rl.data{nBatch}.a(bb)) = rl.data{nBatch}.r(bb) + ...
                                rl.gamma*qsp(idx);
                        end
                    end
                end
                
                if(~isempty(rl.net{lNet}))
                    rl.net{lNet} = trainNetwork(...
                        rl.data{nBatch}.s(:, :, 1, batch),...
                        qsTarget,...
                        rl.net{lNet}.Layers,...
                        rl.trainOpt);
                else
                    rl.net{lNet} = trainNetwork(...
                        rl.data{nBatch}.s(:, :, 1, batch),...
                        qsTarget,...
                        rl.Layers,...
                        rl.trainOpt);
                end
            end
        end
        rl.batchScore = rl.batchScore + sum(rl.reward(sys.Glucose(rl.hist+1:k), 0))/rl.hist;
        
        if(opt.stepPerStep)
            figure(1);
            clf;
            PlotPatientData(sys, 'starttime', dayStart, 'legend', 'northeast')
            subplot(212);
            hold on
            for n = 1:1:length(UBolus)
                if(UBolus(n) > 0)
                    plot(sys.Time(n), 2.5, 'Marker','v','MarkerSize', 10, 'MarkerEdgeColor','g', 'MarkerFaceColor','g');
                    text(sys.Time(n), 2.5, ['    ' num2str(UBolus(n)) ' U'],'Color','r', 'FontSize', 8, 'FontWeight', 'bold');
                end
            end
            disp('press a key ...');
            pause();
        end
        progressbar(sample/rl.batchSize);
    end
    rl.batchScore = rl.batchScore / (sample - 1);
    
    %% plot the last day
    figure(1)
    clf;
    hold on
    PlotPatientData(sys, 'starttime', dayStart, 'legend', 'northeast')
    subplot(211);
    title(sprintf('Results of simulation n%04d: %3.1f%% passive agent, %3.1f%%-greedy', rl.trainingNumber, rl.passive*100, 100-rl.explore*100))
    subplot(212);
    hold on
    for n = 1:1:length(UBolus)
        if(UBolus(n) > 0)
            plot(sys.Time(n), 2.5, 'Marker','v','MarkerSize', 10, 'MarkerEdgeColor','g', 'MarkerFaceColor','g');
            text(sys.Time(n), 2.5, ['    ' num2str(UBolus(n)) ' U'],'Color','r', 'FontSize', 8, 'FontWeight', 'bold');
        end
    end
    if(opt.simMode == 0 && rl.explore < 1e-5)
        print(sprintf('snapshots/snapshot%04d', rl.trainingNumber), '-depsc')
    end
    %% save rl & data set
    if(opt.simMode == 0)
        rl.trainingNumber = rl.trainingNumber + 1;
       
        save('rl.mat','rl')
        save(['backup/rl_' datestr(now,'yymmdd-HHMMSS') '.mat'], 'rl')
        
         if(exist(opt.dataSetName, 'file') == 2)
             load(opt.dataSetName);
             dataSet{end + 1} = rl.data{nBatch};
         else
             dataSet = cell(1);
             dataSet{end} = rl.data{nBatch};
         end
         save(opt.dataSetName, 'dataSet');
    end
    
    %% save scores for benchmarking
    if(opt.simMode > 0 && ~opt.nominalTest)
        if(exist(opt.benchName{opt.simMode}, 'file') == 2)
            load(opt.benchName{opt.simMode});
            score = [score;rl.batchScore];
        else
            score = rl.batchScore;
        end
        save(opt.benchName{opt.simMode}, 'score');
    end
    %% plot batch score
    colormap = {'r', 'g', 'b', 'k'};
    figure(2)
    clf;
    hold on
    plot(1:50, mean(rl.batchScore)*ones(1,50), 'linewidth', 1.5, 'DisplayName', 'Score')
    for bm = 1:length(opt.benchName)
        if(exist(opt.benchName{bm}, 'file') == 2)
            load(opt.benchName{bm});
            plot(1:length(score), (mean(score) + std(score))*ones(1,length(score)), ['--' colormap{bm}], 'linewidth', 1.0)
            plot(1:length(score), (mean(score) - std(score))*ones(1,length(score)), ['--' colormap{bm}], 'linewidth', 1.0)
            plot(1:length(score), (score), ['--' colormap{bm}], 'linewidth', 1.0)
            plot(1:length(score), mean(score)*ones(1,length(score)), 'color', colormap{bm}, 'linewidth', 1.5, 'DisplayName', opt.benchName{bm})
        end
    end
    legend('show')
    grid on
    ylim([0 1])   
    
    disp(rl.batchScore)
end
