classdef MHE_LTI < handle
    %MHE for LTI system
    % author: anas.elfathi@mail.mcgill.ca
    
    properties (GetAccess = public, SetAccess = private)
        N;
        
        A;
        B;
        C;
        D;
        
        Q;
        R;
        
        Y;
        U;
        
        X;
        XLast;
        
        P;
        PLast;
        
        Cons;
        
        NIS;
    end
    
    properties (GetAccess = private, SetAccess = private)
        options;
        Memory = 5e2; % memory
        
        HH;
        HHm;
        fu;
        fy;
        
        n;
        m;
        p;
        
        counter;
    end
    
    methods (Access = public)
        function this = MHE_LTI(model_, conf_)
            this.A = model_.A;
            this.B = model_.B;
            this.C = model_.C;
            this.D = model_.D;
            
            this.n = size(this.A,1);
            this.m = size(this.B,2);
            this.p = size(this.C,1);
            
            this.Reset(conf_);
            
            this.Cons.IneqA = [];
            this.Cons.Ineqb = [];
            this.Cons.EqA = [];
            this.Cons.EqA = [];
            
            this.options = optimoptions('quadprog',...
                'Algorithm','interior-point-convex',...
                'Display','off');
        end
        
        function Reset(this, conf_)
            this.N = conf_.Order;
            
            this.U = NaN*ones(this.m, this.N);
            this.Y = NaN*ones(this.p, this.N);
            
            this.Q = conf_.Q;
            this.R = conf_.R;
            
            this.XLast = conf_.X0;
            this.PLast = conf_.P0;
            
            this.X = zeros(this.n, this.Memory);
            this.X(:, end) = this.XLast;
            this.P = zeros(this.n, this.n, this.Memory);
            this.P(:,:,end) = this.PLast;
            this.NIS = zeros(1, this.Memory);
            
            if(rank(this.Q) < this.n)
                this.Q = this.Q + 1e-8*eye(this.n);
            end
            
            if(rank(this.R) < this.p)
                this.R = this.R + 1e-8*eye(this.n);
            end
            
            this.HH = kron(blkdiag(eye(this.N), 0), this.A'*(this.Q\this.A)) +...
                kron(triu(ones(this.N+1),1)-triu(ones(this.N+1),2), -this.A'/this.Q)+...
                kron(tril(ones(this.N+1),-1)-tril(ones(this.N+1),-2), -this.Q\this.A)+...
                kron(blkdiag(0,eye(this.N)), inv(this.Q) + this.C'*(this.R\this.C));
            
            this.HHm = kron(blkdiag(0,eye(this.N)), - this.C'*(this.R\this.C));
            
            this.fu = kron(eye(this.N+1,this.N), this.A'*(this.Q\this.B))+...
                kron(tril(ones(this.N+1,this.N),-1)-tril(ones(this.N+1,this.N),-2), -this.Q\this.B);
            
            this.fy = kron(tril(ones(this.N+1,this.N),-1)-tril(ones(this.N+1,this.N),-2), -this.C'/this.R);
            
            this.counter = 1;
        end
        
        function ConstraintInit(this, ConsEqA, ConsEqb, ConsIneqA, ConsIneqb)
            this.Cons.EqA = ConsEqA;
            this.Cons.Eqb = ConsEqb;
            this.Cons.IneqA = ConsIneqA;
            this.Cons.Ineqb = ConsIneqb;
        end
        
        function [X_, P_, S_] = Run(this, y, u)
            this.Y(:, 1:end-1) = this.Y(:, 2:end);
            this.Y(:, end) = y;
            
            this.U(:, 1:end-1) = this.U(:, 2:end);
            this.U(:, end) = u;
            
            % Burning first N data
            if(this.counter < this.N)
                P1_0 = this.A*this.PLast*this.A' + this.Q;
                S_ = (this.C*P1_0*this.C' + this.R);
                
                X1_0 = this.A*this.XLast+this.B*u;
                
                if(~isnan(y)) % missing data
                    K = P1_0*(this.C'/S_);
                    P_ = (eye(this.n) - K*this.C)*P1_0;
                    X_ = X1_0 + K*(y - this.C*X1_0);
                else
                   X_ = X1_0;
                   P_ = P1_0;
                end
                % now we start
            else
                % get the most reliable state X0
                X0 = this.X(:, end-this.N+1);
                if(rank(this.P(:, :, end-this.N+1)) < this.n)
                    P0 = this.Q;
                    % reset the covariance
                    this.PLast = 10*this.Q;
                else
                    P0 = this.P(:, :, end-this.N+1);
                end
                
                % predict
                X1_0 = this.A*this.XLast+this.B*u;
                P1_0 = this.A*this.PLast*this.A' + this.Q;
                S_ = this.R + this.C*P1_0*this.C';
              
                % remove missing values from data
                missingIdx = isnan(this.Y);
                Y_ = this.Y(~missingIdx);
                fy_ = this.fy(:,~missingIdx);
                HHm_ = kron(diag([0 missingIdx]), eye(this.n))*this.HHm;
                
                H = kron(blkdiag(1,zeros(this.N)), inv(P0)) +...
                    HHm_ +...
                    this.HH;
                H = (H + H')/2;
                f = kron(eye(this.N+1,1), -P0\X0) +...
                    this.fu*this.U(:) +...
                    fy_*Y_(:);
                %                 r = X0'*(P_\X0) +...
                %                     this.Y(:)' * kron(eye(this.N), inv(this.R)) * this.Y(:) + ...
                %                     this.U(:)'*kron(eye(this.N), this.B'*(this.Q\this.B))*this.U(:);
                
                % forcing the initializiation term to be > 0
                XX_ = [this.X(:, end-this.N+1:end) abs(X1_0)];                
                XX_ = XX_(:);
                
                % only in the first
                if(this.counter == this.N)
                    [XX, ~] = quadprog(...
                        H,...
                        f,...
                        kron(eye(this.N+1),this.Cons.IneqA),...
                        kron(ones(this.N+1,1),this.Cons.Ineqb),...
                        kron(eye(this.N+1),this.Cons.EqA),...
                        kron(ones(this.N+1,1),this.Cons.Eqb),...
                        [],...
                        [],...
                        [],...
                        this.options);
                else
                    [XX, ~] = quadprog(...
                        H,...
                        f,...
                        kron(eye(this.N+1),this.Cons.IneqA),...
                        kron(ones(this.N+1,1),this.Cons.Ineqb),...
                        kron(eye(this.N+1),this.Cons.EqA),...
                        kron(ones(this.N+1,1),this.Cons.Eqb),...
                        min([XX_*0.75 XX_*1.25],[],2),...
                        max([XX_*0.75 XX_*1.25],[],2),...
                        [],...
                        this.options);
                end
                if(~isempty(XX))
                    
                    % only in the first
                    if(this.counter == this.N)
                        this.X(:,end-this.N+1:end) = reshape(XX(1:end-this.n),[this.n,this.N]);
                    end
                    
                    X_ = XX(end-this.n+1:end);
                    
                    % Update covariance of error
                    if(~isnan(y))
                        if(norm(X_ - X1_0) < 1e-5)
                            K = zeros(size(X_));
                        elseif(norm(y - this.C*X1_0) < 1e-5*norm(X_ - X1_0))
                            K = 1e4*ones(size(X_));
                        else
                            K = (X_ - X1_0)/(y - this.C*X1_0);
                        end
                        
                        P_ = (eye(this.n) - K*this.C)*P1_0*(eye(this.n) - K*this.C)' + K*this.R*K';
                    else
                        P_ = P1_0;
                    end
                else                   
                    if(~isnan(y))
                        K = P1_0*(this.C'/S_);
                        P_ = (eye(this.n) - K*this.C)*P1_0;
                        
                        X_ = X1_0 + K*(y - this.C*X1_0);
                    else
                        P_ = P1_0;
                        X_ = X1_0;
                    end
                end
            end
            
            if(sum(isnan(X_(:))) > 0)
                error('Something is wrong');
            end
            this.XLast = X_;
            this.X(:, 1:end-1) = this.X(:, 2:end);
            this.X(:, end) = this.XLast;
            
            if(sum(isnan(P_(:))) > 0)
                error('Something is wrong');
            end
            this.PLast = P_;
            this.P(:, :, 1:end-1) = this.P(:, :, 2:end);
            this.P(:, :, end) = this.PLast;
            
            this.NIS(:, 1:end-1) = this.NIS(:, 2:end);
            this.NIS(:, end) = exp(-(y - this.C*X1_0)'*(S_\(y - this.C*X1_0))/2)/sqrt(2*pi*det(S_));
            
            this.counter = this.counter + 1;            
        end
    end
    
end