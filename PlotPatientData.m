function PlotPatientData( data, varargin )
%% an utility function to plot nice graphics of glucose measurement
% Author: anas.elfathi@mail.mcgill.ca

TStart = 0; %Experiment start time.
figHandle = 1;

dt = 10;
N = length(data.Time);
T = N*dt;
meta = [];
legendPos = 'northwest';

for nVar = 1:2:length(varargin)
    switch(lower(varargin{nVar}))
        case 'meta'
            meta = varargin{nVar+1};
        case 'starttime'
            TStart = varargin{nVar+1};
        case 'figure'
            figHandle = varargin{nVar+1};
        case 'legend'
            legendPos = varargin{nVar+1};
        otherwise
            error('myfuns:PlotPatientData:WrongInput', ...
                'Unkown option %s', varargin{nVar});
    end
end

time_grade = kron(data.Time(2:end), [1;1]);
time_grade(2:end+1) = time_grade;
time_grade(1) = data.Time(1);

figure(figHandle)
clf;

subplot(2,1,1)
if(~isempty(meta))
    title(sprintf('Study ID: %d-%d-%d',meta.StudyID.class,meta.StudyID.patient,meta.StudyID.study));
end
grid on
hold on
plot(data.Time, 10.0*(mod(data.Time + TStart*60, 24*60) < 22*60 & mod(data.Time + TStart*60, 24*60) > 7*60) + 7.0*(mod(data.Time + TStart*60, 24*60) >= 22*60 | mod(data.Time + TStart*60, 24*60) <= 7*60), '--k', 'linewidth', 1)
plot(data.Time, 4.0*ones(size(data.Time)), '--k', 'linewidth', 1)
p1 = plot(data.Time, data.Glucose, '--r*');
CarbsExist = 0;
for n = 1:1:T/dt
    if(data.I_Carbs(n) > 0)
        p2 = plot(data.Time(n), 12, 'Marker','v','MarkerSize', 10, 'MarkerFaceColor',[0.9100    0.4100    0.1700], 'MarkerEdgeColor',[0.9100    0.4100    0.1700]);
        text(data.Time(n), 12, ['    ' num2str(data.I_Carbs(n)) ' g'],'Color',[0.9100    0.4100    0.1700], 'FontSize', 8, 'FontWeight', 'bold');
        CarbsExist = 1;
    end
end
ax = gca;
ax.XTick = 0:2*60:T;
ax.XTickLabel = {mod(TStart:2:TStart+T/60,24)};
xlim([0 T])
ylim([2 max(13, max(data.Glucose)+0.5)])
ylabel('Sensor Glucose (mmol/l)')
xlabel('Time (h)')
if(CarbsExist)
    legend([p1 p2], 'Glucose measurements', 'Meal carbohydrates', 'Location', legendPos)
else
    legend(p1, 'Glucose measurements', 'Location', legendPos)
end
subplot(2,1,2)
grid on
hold on
basal_grade = kron(data.I_basal(1:end-1), [1;1]);
basal_grade(end+1) = data.I_basal(end);
p1 = plot(time_grade, basal_grade, 'b');
BolusExist = 0;
for n = 1:1:T/dt
    if(data.I_bolus(n) > 0)
        p2 = plot(data.Time(n), 2.5, 'Marker','v','MarkerSize', 10, 'MarkerEdgeColor','b', 'MarkerFaceColor','b');
        text(data.Time(n), 2.5, ['    ' num2str(data.I_bolus(n)) ' U'],'Color','b', 'FontSize', 8, 'FontWeight', 'bold');
        BolusExist = 1;
    end
end
ax = gca;
ax.XTick = 0:2*60:T;
ax.XTickLabel = {mod(TStart:2:TStart+T/60,24)};
ylim([0 max(3, max(data.I_basal)+0.5)])
xlim([0 T])
ylabel('Delivred Insulin (U/h)')
xlabel('Time (h)')
if(BolusExist)
    legend([p1 p2], 'Insulin Basal', 'Insulin Bolus', 'Location', legendPos)
else
    legend(p1, 'Insulin Basal', 'Location', legendPos)
end

drawnow;
end

