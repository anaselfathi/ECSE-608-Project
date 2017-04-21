classdef FIR_Filter < handle
    %FIR_FILTER a handy filter
    %   for each step time n
    %   y(n) = a(1)*y(n-1) + a(2)*y(n-2) + ... + a(N)*y(n-N) +...
    %   b(1)*u(n) + b(2)*u(n-1) + ... + b(M)*u(n-M+1) + b(M+1)*u(n-M)
    %   length of a deermines the history of y
    %   length of b determines the history of u
    %   DC gain is 
    %         b(1) + b(2) + ... + b(M+1)
    %   DC = ----------------------------
    %         1 - a(1) - ... - a(N)
    % Author: anas.elfathi@mail.mcgill.ca
  
    properties (GetAccess = public, SetAccess = private)
        N = 1;             % order of FIR
        M = 1;             % order of FIR
        coeff = [1/2;1/2]; % list of coefficient of the FIR filter
        y = 0;             % data history
        u = 0;
    end
    
    methods (Access = public)
        function this = FIR_Filter(a_, b_, y0)
            a_ = a_(:);            
            b_ = b_(:);
            this.N = length(a_);
            this.M = length(b_)-1;
            
            this.coeff = [a_;b_]; % size is N+M+1
            
            this.u = zeros(1, this.M);
            if(isempty(y0))
                this.y = zeros(1, this.N);
            else
                if(length(y0) == N)
                    error('initialisation data should be equal to FIR order');
                else
                    this.y = y0(:)';
                end
            end
        end
        
        function y_ = Update(this, u_)
            % filter
            data = [this.y, u_, this.u];
            y_ = data * this.coeff;
            
            % save old data
            this.y(2:end) = this.y(1:end-1);
            this.y(1) = y_;

            if(~isempty(this.u))
                this.u(2:end) = this.u(1:end-1);
                this.u(1) = u_;
            end
        end
    end
    
end

