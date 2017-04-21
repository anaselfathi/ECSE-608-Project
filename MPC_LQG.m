classdef MPC_LQG < handle
    %MPC for LTI system with quadratic cost
   % Author: anas.elfathi@mail.mcgill.ca
 
    properties (GetAccess = public, SetAccess = private)
        N;
        
        A;
        B;
        C;
        D;
        
        Gy;
        Gyf;
        Gu;
    end
    
    properties (GetAccess = private, SetAccess = private)
        options;
        
        AA;
        BB;
        CC;
        DD;
        
        Gyy;
        Guu;
    end
    
    methods (Access = public)
        function this = MPC_LQG(model_, conf_)
            this.A = model_.A;
            this.B = model_.B(:,1);
            this.C = model_.C;
            this.D = model_.D;
            
            this.N = conf_.NStep;
            
            this.Gy = conf_.Gy;
            this.Gyf = conf_.Gyf;
            this.Gu = conf_.Gu;
            
            this.Gyy = diag(horzcat(this.Gy*ones(1, this.N-1), this.Gyf));
            this.Guu = diag(this.Gu*ones(1, this.N));
            
            this.GetAugmentedStateSpace();
            
            this.options = optimoptions('quadprog',...
                'Algorithm','interior-point-convex','Display','off');
        end
        
        
        function Reset(this, conf_)
            this.Gy = conf_.Gy;
            this.Gyf = conf_.Gyf;
            this.Gu = conf_.Gu;
            
            this.Gyy = diag(horzcat(this.Gy*ones(1, this.N-1), this.Gyf));
            this.Guu = diag(this.Gu*ones(1, this.N));
        end
        function GetAugmentedStateSpace(this)
            
            n = size(this.A,1);
            m = size(this.B,2);
            
            this.AA = zeros(n*this.N, n);
            this.BB = zeros(n*this.N, m*this.N);
            P = eye(n);
            for p = 1:this.N
                if(p > 1)
                    this.BB((p-1)*n+1:(p)*n, m+1:end) = this.BB((p-2)*n+1:(p-1)*n, 1:(this.N-1)*m);
                end
                this.BB((p-1)*n+1:(p)*n, 1:m) = P*this.B;
                P = P*this.A;
                this.AA((p-1)*n+1:(p)*n,:) = P;
            end
            this.CC = kron(eye(this.N,this.N), this.C);
            
        end
        
        function [U, J] = Run(this, X0_, Yr_, Ymin_, Umin_, Umax_, U0_)
            
            N_ = this.N;
            if(N_ ~= length(Yr_))
                U = 0;
                J = inf;
                return;
            end
            
            AA_ = this.AA;
            BB_ = this.BB;
            CC_ = this.CC;
            Gyy_ = this.Gyy;
            Guu_ = this.Guu;
            
            H = BB_'*CC_'*Gyy_*CC_*BB_ + Guu_;
            f = ((X0_'*AA_'*CC_' - Yr_')*Gyy_*CC_*BB_)';
            r = 0.5 * (CC_*AA_*X0_ - Yr_)'*Gyy_*(CC_*AA_*X0_ - Yr_);
            
            H = (H + H')/2;
            
            [UU, c] = quadprog(...
                H,...
                f,...
                -CC_*BB_,...
                CC_*AA_*X0_ - Ymin_,...
                [],...
                [],...
                Umin_*ones(N_,1),...
                Umax_*ones(N_,1),...
                U0_*ones(N_,1),...
                this.options);
            
            if(~isempty(UU))
                U = UU(1);
                J = c + r;
            else
                U = 0;
                J = r;
            end
            
        end
    end
    
end