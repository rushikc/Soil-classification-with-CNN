#python code to process initial data

import xlrd
import xlwt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_excel('data.xlsx')

data.corr()

names = data.columns

data.drop_duplicates(inplace=True)  # removing duplicates

data = data.dropna(how='any')

writer=pd.ExcelWriter('data_nona.xlsx')
data.to_excel(writer,'Sheet1')
writer.save()

limit = {}


def IQR_outlier(dt, name):
    q1 = dt.quantile(.25)
    q3 = dt.quantile(.75)
    iqr = q3 - q1
    l_limit = q1 - 1.5 * iqr
    r_limit = q3 + 1.5 * iqr
    l_limit = round(l_limit, 2)
    r_limit = round(r_limit, 2)
    limit[name] = [l_limit, r_limit]


def removal_outlier(st, name):
    st = st[st < limit[name][1]]
    st = st[st > limit[name][0]]
    return st


for i in range(len(names)):
    IQR_outlier(data[names[i]], names[i])

for i in range(len(names)):
    data[names[i]] = removal_outlier(data[names[i]], names[i])

soil = data['soil_type'].values.tolist()
ph = data['ph'].values.tolist()
p = data['av_p'].values.tolist()
k = data['av_k'].values.tolist()
s = data['av_s'].values.tolist()
zn = data['av_zn'].values.tolist()


from scipy.stats import zscore

data.apply(zscore)  # feature scaling

from sklearn.preprocessing import MinMaxScaler
data_minmax=(d_input-d_input.min()/(d_input.max()-d_input.min()))
(d_input-d_input.min()/(d_input.max()-d_input.min())) #min-max normalization

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.25)  # splitting data

limit={}
limit['RED SANDY']=1
limit['BLACK SOIL']=1
limit['SANDY SOIL']=1
limit['others']=1
limit['RED SOIL']=1

for i in range(soil.__len__()):
    st = str(soil[i]).upper()
    if st.__contains__('RED SANDY'):
        limit[st]+=1
    elif st.__contains__('RED SOIL'):
        limit[st]+=1
    elif st.__contains__('BLACK SOIL'):
        limit[st] += 1
    elif st.__contains__('SANDY SOIL') or st.__contains__('SANDY LOAM'):
        limit['SANDY SOIL'] += 1
    else:
        soil[i] = 'others'
        limit['others']+=1

df = pd.DataFrame()
for column in data.columns[10:]:
    df[column] = data[column]



df1 = pd.DataFrame({'col':soil})
df['soil'] = df1


del df['authority']
del df['av_b']
del df['av_fe']
del df['av_cu']
del df['av_mn']

df.dropna()

train.to_excel('train.xlsx')
test.to_excel('test.xlsx')

train_cat.to_excel('train_cat.xlsx')
test_cat.to_excel('test_cat.xlsx')


writer=pd.ExcelWriter('data_ulti.xlsx')
d1.to_excel(writer,'Sheet1')
writer.save()

d_input=d1.iloc[:,0:4]
d_output=d1.iloc[:,4:]

df.to_excel('data1.xlsx')



################################################################


function [Y,Xf,Af] = myNeuralNetworkFunction(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
%
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = 4xQ matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = 4xQ matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%

% NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [6.3;0.18;17;2];
x1_step1.gain = [0.8;1.98019801980198;0.0350877192982456;0.0820344544708778];
x1_step1.ymin = -1;

% Layer 1
b1 = [1.8121688896422643023;-2.9153479903258281425;-0.5030964366729473447;-0.9403050003828167247;-5.8087978053479512042;2.1259111533313741482;3.8760147602780290477;0.46484904026185080994;-0.2640559421507817639;1.1727797340158472306;2.6989865051305876875;4.7387353961300329175;2.5428422713606710914;-2.8068751352825880296;-3.4690407447051736511;0.8892443800078153604;0.98634277094409883446;1.5151753030701644587;-0.95962193609870150013;-0.64595201427146564654;-0.30213587903995842154;3.0384724779283085461;0.50657345606061821019;7.2867970744961256813;6.9940435693821116914];
IW1_1 = [6.8686045776431381427 2.7750435842134817399 1.3639466623261915501 1.1228445025279103486;2.5866547672475648234 -3.1914466971735002687 -3.5262126980964518808 -1.7632505626653072195;1.7934677413842976534 -1.602071415071244509 -0.37809689991647232876 0.12212188811142166145;-1.018906795976839863 2.193851759132690038 -2.9658576152971258821 -4.9520781652883929524;-9.1785320600441604455 0.40420489924474323207 7.3538756807084064704 5.5158851124206957905;-9.5197701776829699583 -1.2083084902211826961 -1.8534414334413913394 2.091933090936787476;-9.152503456081399591 -3.8538177620451170569 6.1347857353802961455 0.72242642270097745527;14.848578869501254829 9.6071576113915053696 -6.5250293482871404294 5.942986849238326208;-2.8345678562345835161 2.5551616255203528638 -2.788825809032566827 4.1513819650848100551;-5.9762379478750800033 -6.7400940281723356051 -0.38471564489127063613 -3.0048717764126937624;8.742426674402667075 4.7141117727029318019 1.8668872732945274162 1.5855093525648662478;-0.67496826917988717032 3.8380288241962934315 -9.3913692951960978661 -2.1474972898350119799;-8.0337041944938913929 -0.43983988750849961624 -1.6416806874068210487 2.4358841626781231327;7.2159163144359697029 3.3787520757435629548 -4.4931950626602521481 -0.35906106049383296197;6.5472221314753058508 -6.8566547154360915073 -2.137682373646839995 5.2294729895393050612;-0.44293649406270485525 12.498474924019308929 -1.0683079454657211649 2.438161239349852849;6.0961113919656444438 1.1872971202901823062 0.81162490655508900961 0.83805786075654176148;-4.8874740249305483886 5.5632814651250654947 -3.6565917139931980451 -6.8930971453931135784;11.368494141655315133 -3.927817414872506685 -1.0879268705461195843 3.0104095249511164667;4.057339426614258393 -4.0595669387559443209 3.2271896529575814228 6.3511673398837427129;-2.7260621130866820039 1.1768251077910540925 -2.4488946583256279155 5.365827319412309393;1.0724970512470604067 -4.1952054966767740041 -0.055136020761241764931 -0.4840478692664431648;-0.71021331706518298077 0.73692716236126443174 0.12057694593894337232 0.012965472250712361049;8.7731637847050087942 -1.3287944748011364382 -9.1348957853997170275 -4.6182783243200109524;-1.8889428235435783421 5.6308446186210678874 2.5371890323438508474 5.2419109445898071087];

% Layer 2
b2 = [-2.2018615934286263069;2.1823217745996918993;-0.97846769411326617316];
LW2_1 = [-5.1962183383050026819 -0.48540824652578568044 2.0352907120920908923 -0.69837340219420118714 -2.609016835147961455 2.750637608936677303 2.3551979638932509786 -0.65706301681564871497 1.5524350775782875811 -0.50797592351074327688 3.2538667881197791409 0.48216451506702406871 -2.6965973418715916132 3.1095881917997969524 0.47377882104620067105 -0.5628055023476120633 3.3291985030133139922 -2.5223805784123847218 -0.39957458451071237171 -2.684923181396367653 -1.7256592123907827929 1.5804006580554461614 4.5830510807540978391 -2.5465930011708430847 -0.50835924552187117254;5.2055664179557625815 0.47537759901369924798 -2.0323586728804388102 0.69857082236165901623 2.5470236497225271499 -2.7881885342012826534 -2.4008674862936372207 0.66298611137956053874 -1.5503128563374082294 0.50139233444081043878 -3.2617767637415444604 -0.47862112405291395989 2.7252374821239269131 -3.1651572189679151315 -0.47410405941012756514 0.56470043060973684756 -3.3596345345887952405 2.5982527935681098974 0.41403446858604014968 2.7372795886094309914 1.7207048932091200122 -1.6129587815407118168 -4.6299965439796872957 2.5246024882740067952 0.51477353500354117166;-0.0092132263230420077815 0.010141542087720792353 -0.0040230633951898490605 6.6099709550425447892e-06 0.062240536617375961936 0.037175603553662998901 0.0453889169434931114 -0.0059091280042287055302 -0.0025024909207903992328 0.0066495022643623061948 0.0078080189000762998536 -0.0036424805700115865659 -0.028306706431981916439 0.055242320216520181264 0.00010221681207623503173 -0.0017919885477168141785 0.030257759256403341658 -0.075583963828098288396 -0.014386829854962378955 -0.05200131784779061378 0.0055380941940572827459 0.031215269356369028386 0.044299700282799961415 0.022084715034917060117 -0.0062320673754139950182];

% Output 1
y1_step2.ymin = -1;
y1_step2.gain = [2;2;2];
y1_step2.xoffset = [0;0;0];
y1_step1.xrows = 4;
y1_step1.keep = [1 2 3];
y1_step1.remove = 4;
y1_step1.constants = 0;

%SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
    X = {X};
end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
    Q = size(X{1},2); % samples/series
else
    Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS
    
    % Input 1
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = repmat(b2,1,Q) + LW2_1*a1;
    
    % Output 1
    temp = mapminmax_reverse(a2,y1_step2);
    Y{1,ts} = removeconstantrows_reverse(temp,y1_step1);
end

% Final Delay States
Xf = cell(1,0);
Af = cell(2,0);

% Format Output Arguments
if ~isCellX
    Y = cell2mat(Y);
end
end


%MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end

% Remove Constants Output Reverse-Processing Function
function x = removeconstantrows_reverse(y,settings)
Q = size(y,2);
x = nan(settings.xrows,Q,'like',y);
x(settings.keep,:) = y;
x(settings.remove,:) = repmat(settings.constants,1,Q);
end

