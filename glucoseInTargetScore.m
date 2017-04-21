function s = glucoseInTargetScore(g, meanVal)
%% glucoseInTargetScore
% provides a score on how well the glucose levels are close to target
% score = 1 if all is good
% Author: anas.elfathi@mail.mcgill.ca

if(nargin < 2)
    meanVal = 1;
end

% score param
persistent param
if(isempty(param))
    param.target = 5.5;
    param.min = 4.0;
    param.max = 10.0;
    
    param.gain = 1/log((param.max - param.target)/(param.target - param.min));
    
    param.alp = (exp(1/param.gain) - exp(-1/param.gain))/(param.max - param.min);
    param.bet = (param.max*exp(-1/param.gain) - param.min*exp(1/param.gain))/(param.max - param.min);
end


s  = 1 - 0.9*param.gain*abs(log(param.alp*g + param.bet));

s(s < -1.0 & g > param.target) = -1;
s(s < -1.5 & g < param.target) = -1.5;

if(meanVal)
    s = mean(s);
end

end