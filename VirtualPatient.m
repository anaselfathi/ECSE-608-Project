classdef VirtualPatient < handle
    %VirtualPatient a Type-1 Diabetes virtual patient simulation
    %  type is 0: Minimal - 1: Hovroka's
    % Author: anas.elfathi@mail.mcgill.ca
  
    properties (GetAccess = public, SetAccess = private)
        type;  % type is 0: Minimal - 1: Hovroka's 2: Cobelli 3: Experrimental
        
        X; % state
    end
    
    properties (GetAccess = private, SetAccess = private)
        %model default parameters
        min; % minimal model parameters 0
        hov; % hovorka's model parameters 1
        
        meal;
        flux;
        noise;
    end
    
    methods (Access = public)
        function this = VirtualPatient(type_)%, [])
            
            this.type = type_;
            
            this.initHovorkaProp();
            this.initMinimalProp();
            
            this.reset();
            
            this.noise.process = 0;
            this.noise.sensor = 0;
            this.noise.sFIR = FIR_Filter([0.75 0.2], [0.05], []);
            this.noise.pFIR = FIR_Filter([0.5 0.2], [1 -0.7], []);
            
            this.X = this.getX0();
        end
        
        function setNoise(this, process, sensor)
            if(~isempty(process))
                this.noise.process = process;
            end
            if(~isempty(sensor))
                this.noise.sensor = sensor;
            end
        end
        
        function reset(this)
            this.meal.index = 0;
            this.meal.added = 0;
            this.meal.value = [];
            this.meal.time = [];
            
            mu_i = 0;
            sigma_i = log((1+0.07/6)/(1-0.07/6))/0.75;
            i_flux = normrnd(mu_i, sigma_i, 1, 24);
            i_flux = cumsum(i_flux, 2);
            i_flux = exp(i_flux);
            i_flux = i_flux ./ geomean(i_flux, 2);
            this.flux.ImDay = i_flux;
            
            this.flux.FmDay = zeros(1,24);
            
            % linear polynomial interpolation
            x = 0:1:23;
            g = [0, 23; 2, -2];
            poly_g = polyfit(g(1,:), g(2,:), 1);
            func_g = poly_g(1)*x + poly_g(2);
            mu_g = 0;
            sigma_g = 1.7/6 / .75;
            g_flux = normrnd(mu_g, sigma_g, 1, 24);
            g_flux = cumsum(g_flux, 2);
            g_flux = g_flux - mean(g_flux, 2);
            g_flux = g_flux + func_g;
            this.flux.FgDay = g_flux;
            
        end
        
        function P = getPatientProp(this)
            switch(this.type)
                case 1
                    P = this.hov;
                otherwise
                    P = this.min;
            end
        end
        
        function setPatientProp(this, P)
            switch(this.type)
                case 1
                    this.hov = P;
                otherwise
                    this.min = P;
            end
        end
        
        function setMinimalProp(this, Weight, ICR, TDD, TDB)
            this.min.weight = Weight; %kg
            this.min.TDD = TDD; % total daily dose (Units)
            this.min.TDB = TDB; % total daily dose (Units)
            this.min.MCHO = 180.156; % molar mass for glucose g/mol
            this.min.ICR = ICR; % Units / g
            this.min.Bio = 1.0;
            this.min.MCHO = 180.156; % molar mass for glucose g/mol
            this.min.Vg = 0.16; % Distrubition volume of the measurement accessible compartiment L/kg
            this.min.Si = 1e3*this.min.Bio/(this.min.weight*this.min.MCHO*this.min.ICR*this.min.Vg); % nominal sensivity level mmol/L /Units
            this.min.P1 = 1e-3; % glucose effect on itself
            this.min.EGP0 = this.min.Si*TDB/(24*60); % mmol per L per min
            this.min.Tau_i = 30; % min
            this.min.Tau_m = 24; % min
            this.min.Tau_sen = 15; % min
        end
        
        function X0 = getX0(this)
            switch(this.type)
                case 1
                    X0 = this.getHovorkaX0();
                otherwise
                    X0 = this.getMinimalX0();
            end
        end
        
        function G = getGlucose(this)
            switch(this.type)
                case {1, 4}
                    G = this.X(end);
                otherwise
                    G = this.X(1);
            end
            if(this.noise.sensor > 0)
                G = G + this.noise.sFIR.Update(randn(1)*this.noise.sensor);
            end
        end
        
        % basal in Units/min
        % bolus Units
        % meal grams
        % time is an interval
        function X = model(this, X0, basal, bolus, meal, time)
            if(mod(time(1), 24*60) == 0)
                this.reset();
            end
            
            switch(this.type)
                case 1 % Hovorka's model
                    % add meal
                    if meal ~= 0
                        if(~this.meal.added)
                            this.meal.index = this.meal.index + 1;
                            this.meal.value(this.meal.index) = meal;
                            this.meal.time(this.meal.index) = time(1);
                            this.meal.km(this.meal.index) = this.hov.km*exp(.1*randn(1));
                            this.meal.km(this.meal.index) = max(min(this.meal.km(this.meal.index), 1.5*this.hov.km), 0.5*this.hov.km);
                            this.meal.added = 1;
                        end
                    else
                        this.meal.added = 0;
                    end
                    % add bolus
                    X0(1) = X0(1) + bolus;
                    % simulate ode45
                    [~,Y_] = ode45(@(t_,y_) this.hovorka(t_, y_, basal),...
                        time,...
                        X0);
                    X = Y_(end,:);
                otherwise % Minimal
                    % add meal
                    X0(7) = X0(7) + this.min.Bio * 1e3 * meal / (this.min.weight * this.min.MCHO);
                    % add bolus
                    X0(5) = X0(5) + bolus;
                    % simulate
                    X = this.minimal(time, basal, X0);
            end
            
            X = X(:);
            this.X = X;
        end
        
    end
    
    methods (Access = private)
        %% Hovoka
        function initHovorkaProp(this)
            % init hovrka's model parameter
            this.hov.kis = .02; %(l/min)
            this.hov.ke = .076; %(l/min)
            this.hov.ci = .10/60; %(U/min)
            this.hov.weight = 45; %(kg)
            this.hov.ka1 = 4*10^(-2); %(l/min)
            this.hov.ka2 = 5.9*10^(-2); %(l/min)
            this.hov.ka3 = 6.3*10^(-2); %(l/min)
            this.hov.st = 11.4*10^(-4); %(/min/mU/l)
            this.hov.sd = 3.36*10^(-4); %(/min/mU/l)
            this.hov.se = 117*10^(-4); %(/mU/l)
            this.hov.f01 = 7.3; %(umol/kg/min)
            this.hov.k12 = 8.6*10^(-2); %(l/min)
            this.hov.EGP0 = 26.3; %(umol/kg/min)
            this.hov.d = 9.6;%(min)
            this.hov.km = .025; %(/min)
            this.hov.pm = .8 ; %(unitess)
            this.hov.k = 0.0614;
            this.hov.Vi = 190; %(ml/kg)
            this.hov.V = 160; %(ml/kg)
            this.hov.TDD = 70; % total daily dose (Units)
            this.hov.ICR = 1/8; % Units / g
        end
        
        function X0 = getHovorkaX0(this)
            Gs0 = 5.5;
            
            % roots([-hov.EGP0*hov.se*hov.sd-hov.st*hov.sd*Q10, -hov.f01*((Q10/160)/(1+Q10/160))*hov.sd-hov.EGP0*hov.se*hov.k12+hov.EGP0*hov.sd-hov.st*Q10*hov.k12 + hov.k12*Q10*hov.st*hov.k12, hov.k12*hov.EGP0-hov.k12*hov.f01*(Q10/160)/(1+Q10/160)])
            % 0.55*hov.TDD/(24*60)
            Ub = 1.62/60; %(U/min)
            Qis1 = Ub/this.hov.kis; %(U)
            Qis2 = Qis1; %(U)
            Qi0 = (Ub + this.hov.ci)/this.hov.ke; %(U)
            Ip = Qi0 * 10^6 / (this.hov.Vi * this.hov.weight); %(mU/l)
            
            x10 = Ip; %(mU/l)
            x20 = Ip; %(mU/l)
            x30 = Ip; %(mU/l)
            
            Q10 = Gs0*this.hov.V; %(umol/kg)
            Q20 = Q10*x10*this.hov.st/(x20*this.hov.sd+this.hov.k12); %(umol/kg)
            
            X0 = [Qis1;...
                Qis2;...
                Qi0;...
                x10;...
                x20;...
                x30;...
                Q10;...
                Q20;...
                Gs0];
            
            this.X = X0;
        end
        
        function dydt = hovorka(this, t, y, basal_insulin)
            
            Qis1 = y(1); % Units
            Qis2 = y(2); % Units
            Qi = y(3);   % Units
            x1 = y(4);   % 1 / (10?4 × /min per mU/l)
            x2 = y(5);   % 1 / (10?4 × /min per mU/l)
            x3 = y(6);   % 1 / (10?4 per mU/l)
            Q1 = y(7);   % ?mol/kg
            Q2 = y(8);   % ?mol/kg
            Gs = y(9);   % mmol/l
            
            dydt = zeros(9,1);
            
            % subcutaneous insulin absorption subsystem
            dydt(1) = basal_insulin/60 - Qis1 * this.hov.kis;
            dydt(2) = Qis1 * this.hov.kis - Qis2 * this.hov.kis;
            
            % plasma insulin kinetics subsystem
            dydt(3) = Qis2 * this.hov.kis - Qi * this.hov.ke + this.hov.ci;
            
            Ip = Qi * 10^6 / (this.hov.Vi * this.hov.weight); %(mU/l)
            
            % insulin action subsystem
            dydt(4) = -this.hov.ka1 * x1 + this.hov.ka1 * Ip;
            dydt(5) = -this.hov.ka2 * x2 + this.hov.ka2 * Ip;
            dydt(6) = -this.hov.ka3 * x3 + this.hov.ka3 * Ip;
            
            % gut absorption subsystem
            Um = 0;
            if this.meal.index > 0
                for m = 1:this.meal.index
                    Um1 = this.meal.km(m)^2 * (t-this.meal.time(m)) * exp(-this.meal.km(m)*(t-this.meal.time(m)))*(this.meal.value(m)*5551)/this.hov.weight * this.hov.pm;
                    if t > this.meal.time(m) + this.hov.d
                        Um2 = this.meal.km(m)^2 * (t-this.meal.time(m)-this.hov.d) * exp(-this.meal.km(m)*(t-this.meal.time(m)-this.hov.d))*(this.meal.value(m)*5551)/this.hov.weight * (1-this.hov.pm);
                    else
                        Um2 = 0;
                    end
                    Um = Um + (Um1 + Um2);
                end
            end
            
            % glucose kinetics subsystem
            if x3*this.hov.se < 1
                dydt(7) = -this.hov.f01 * (Q1/160)/(1+Q1/160) - x1*this.hov.st*Q1 + this.hov.k12*Q2 + this.hov.EGP0*(1-x3*this.hov.se) + Um;
                dydt(8) = x1*this.hov.st * Q1 - (this.hov.k12+x2*this.hov.sd) * Q2;
            else
                dydt(7) = -this.hov.f01 * (Q1/160)/(1+Q1/160) - x1*this.hov.st*Q1 + this.hov.k12*Q2 + Um;
                dydt(8) = x1*this.hov.st * Q1 - (this.hov.k12+x2*this.hov.sd) * Q2;
            end
            
            Gt = Q1 / this.hov.V; %(mmol/l)
            
            % glucose sensor
            dydt(9) = this.hov.k*Gt - this.hov.k*Gs;
        end
        
        %% Minimal
        function initMinimalProp(this)
            this.min.weight = 45; %kg
            this.min.TDD = 70; % total daily dose (Units)
            this.min.TDB = 0.55*this.min.TDD; % total daily dose (Units)
            this.min.MCHO = 180.156; % molar mass for glucose g/mol
            this.min.ICR = 0.1; % Units / g
            this.min.Bio = 1.0;
            this.min.MCHO = 180.156; % molar mass for glucose g/mol
            this.min.Vg = 0.16; % Distrubition volume of the measurement accessible compartiment L/kg
            this.min.Si = 1e3*this.min.Bio/(this.min.weight*this.min.MCHO*this.min.ICR*this.min.Vg); % nominal sensivity level mmol/L /Units
            this.min.EGP0 = this.min.Si*this.min.TDB/(24*60); % mmol per L per min
            this.min.Tau_i = 30; % min
            this.min.Tau_m = 24; % min
            this.min.Tau_sen = 15; % min
            
            Si = this.min.Si;
            Tau_i = this.min.Tau_i;
            Tau_m = this.min.Tau_m;
            Tau_sen = this.min.Tau_sen;
            Vg = this.min.Vg;
            
            A = ...
                [-1/Tau_sen, 1/(Tau_sen), 0 , 0        , 0       , 0           , 0       ;...
                0          , 1e-8       , 1 , -Si/Tau_i, 0       , 1/(Vg*Tau_m), 0       ;...
                0          , 0          , 1e-8 , 0        , 0       , 0           , 0       ;...
                0          , 0          , 0 , -1/Tau_i , 1/Tau_i , 0           , 0       ;...
                0          , 0          , 0 , 0        , -1/Tau_i, 0           , 0       ;...
                0          , 0          , 0 , 0        , 0       , -1/Tau_m    , 1/Tau_m ;...
                0          , 0          , 0 , 0        , 0       , 0           , -1/Tau_m];
            
            B = ...
                [0;...
                0;...
                0;...
                0;...
                1/60;...
                0;...
                0];
            
            this.min.model.A = A;
            this.min.model.B = B;
            
            A = this.min.model.A;
            B = this.min.model.B;
            
            dt = 10;
            
            K = ...
                [0;...
                0;...
                1e-4/this.min.EGP0;...
                0;...
                0;...
                0;...
                0];
            
            Ad = expm(A*dt);
            Bd = ((Ad - eye(size(A)))/(A))*B;
            Kd = ((Ad - eye(size(A)))/(A))*K;
            
            this.min.model.Ad = Ad;
            this.min.model.Bd = Bd;
            this.min.model.Kd = Kd;
        end
        
        function X0 = getMinimalX0(this)
            Gs0 = 5.5;
            this.min.Ub = 0.55*this.min.TDD/(24); %(U/min)
            
            X0 = [Gs0;...             % sensor glucose  mmol / L
                Gs0;...               % plasma glucose  mmol / L
                this.min.EGP0;...     % EGP               mmol / L / min
                this.min.Ub*this.min.Tau_i/60;... % plasma insulin mass Units
                this.min.Ub*this.min.Tau_i/60;... % first compartiment insulin mass Units
                0;...                 % meal-glucose second comp
                0];
            
            this.X = X0;
        end
        
        function X = minimal(this, t, u, X)
            X =  this.min.model.Ad*X +  this.min.model.Bd*u;
            
            if(this.noise.process)
                X = X +  this.min.model.Kd*this.noise.pFIR.Update(randn(1)*this.noise.process);
            end
        end
    end
end