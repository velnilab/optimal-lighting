#Importing libraries
#############################################################################################
import statistics                                        #for calculating mean
import RPi.GPIO as IO                                    #to generate PWM
import time                                              #for delay
import board                                             #to create I2C bus
import busio                                             #to create I2C bus
import adafruit_ads1x15.ads1115 as ADS                   #ads1115
from adafruit_ads1x15.analog_in import AnalogIn          #reading the analog input
import numpy as np                                      
import math 
import pandas as pd                                      # for reading the xls file
import cvxpy as cp                                       #for solving the optimization problem 
import matplotlib.pyplot as plt    
from datetime import datetime                            #for the date

#############################################################################################
#############################################################################################
#Starting hour and date

now= datetime.now()
month=int(now.strftime("%m"))
day=int(now.strftime("%d"))
year=int(now.strftime("%Y"))
hour=int(now.strftime("%H"))
minuite=int(now.strftime("%M"))
second=int(now.strftime("%S"))

print('date:',now)
date = np.array([[month,day,year],[hour,minuite,second]])
np.savetxt('day_'+str(month)+'_'+str(day),date)           #saving date in Raspbeery Pi for future records

#############################################################################################
#############################################################################################
#PWM set-up
 
IO.setmode(IO.BCM)
IO.setwarnings(False)
IO.setup(18,IO.OUT)                                       #GPIO 18 for prediction_based method
p_p = IO.PWM(18,1000)                                     #f=1000hz
IO.setup(13,IO.OUT)                                       #GPIO 13 for heuristic method
p_h = IO.PWM(13,1000)                                     #f=1000hz 
p_p.start(0)                                              #Initial duty cycle=0
p_h.start(0)                                              #Initial duty cycle=0

#############################################################################################
#############################################################################################
# reading Markov data from excel files based on month

data1 = pd.read_excel("/home/pi/exp/m_"+str(month)+".xls",sheet_name='Sheet1',header=None) 
data2 = pd.read_excel("/home/pi/exp/m_"+str(month)+".xls",sheet_name='Sheet2',header=None) 
data3 = pd.read_excel("/home/pi/exp/m_"+str(month)+".xls",sheet_name='Sheet3',header=None) 
data4 = pd.read_excel("/home/pi/exp/m_"+str(month)+".xls",sheet_name='Sheet4',header=None)
data5 = pd.read_excel("/home/pi/exp/m_"+str(month)+".xls",sheet_name='Sheet5',header=None)
data6 = pd.read_excel("/home/pi/exp/m_"+str(month)+".xls",sheet_name='Sheet6',header=None)

Mean_Z = np.array(data1).reshape((1,-1))                  #mean solar irradiance in each state (1*48 matrix)
mean_daily_data = np.array(data2).reshape((-1,1))         #mean solar irradiance for a day (needed for before sunrise) ((prediction_time*4)*1 matrix)
TM = np.array(data3)                                      #transition matrix for the 4 zones (48*48 matrix)
Z_All = np.array(data4)                                   #irradiance value of each state in each zone (4*13 matrix)
data5 = np.array(data5)                                   #changing panda data frame to numpy array
prediction_time = int(data5)                              #number of hours that sunlight needs to be predicted (different for each month)
Z_Max_All = np.array(data6).reshape((1,-1))               #The max reasonable irradiance amount in each zone (1*4 matrix)//use reshape to change vector to matrix, row ones

global S,k,a,C,D_n,U_LED,steps
def frange(start, stop, step):                             #range function for float numbers 
    i = start
    while i < stop:
        yield i
        i += step
t = list(frange(4.5,20.5,0.25))                             #Define initial and terminal hour of photoperiod here/ Each time step is 15 mins. 
conversion_co_train = 2.02*0.4                              #2.02 for changing sunlight irradiance to PPFD
conversion_co_test = 0.4                                    #40% of sunlight reaches inside of a greenhouse
n = 12                                                      #number of states in each zone
zone_num = 4                                                #number of zones
q = 3                                                       #original solar data was 12 samples per hour, but we need 4 samples per hour
a = 121                                                     #constant in ETR, PPFD relationshhip
k = 0.00277                                                 #constant in ETR, PPFD relationshhip
U_LED = 86.21                                               #max ETR of LED (PPDF=450)
lb_etr = 14.18                                              #convert 10% of the max PPFD to ETR
iterations = int(12/q*16)                                   #number of iterations for a 16 hour photoperiod
# Variable Electricity prices 
C11 = np.array(([1.5,1.5,1.5,1.5,1.8,1.8,1.8,1.8,2.5,2.5,2.5,2.5,3,3,3,3,2.7,2.7,2.7,2.7,2.6,2.6,2.6,2.6,2.7,2.7,2.7,2.7,2.6,2.6,2.6,2.6,2.5,2.5,2.5,2.5,2.6,2.6,2.6,2.6,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.9,2.9,2.9,2.9,3.9,3.9,3.9,3.9,3.4,3.4,3.4,3.4,3.1,3.1,3.1,3.1])).reshape((1,-1))
C1 = 5.23*C11                                               #each entry multiples by 5.23
# Constant Electricity prices 
#C1 = 13.19*np.ones((1,iterations))
C = C1
X_Light = np.zeros((iterations,1))                          #supplemental lighting in prediction method
X_Light1 = np.zeros((iterations,1))                         #supplemental lighting in heuristic method
S1 = np.zeros((iterations,1))                               #predicted sunlight
Sunlight_Until_Now = np.zeros((iterations,1))
xxx = np.zeros((iterations,iterations))
counter = 1
samples_without_sunlight = int((16-prediction_time)*12/q)   #before sunrise and after sunset 

#############################################################################################
#############################################################################################
# Solving the OP Before sunrise

nosunlight = np.zeros((int(samples_without_sunlight/2),1))  #preparing S_mean_real as S1
S_Test = np.zeros((iterations,1))                           #real sunlight(S_sensor)
S_mean_real = np.concatenate((nosunlight,mean_daily_data,nosunlight), axis=0) 
S_mean_real_PPFD = np.zeros((len(S_mean_real),1))
for i in range(0,S_mean_real.shape[0]):                     #convert irradiance to PPFD
    S_mean_real_PPFD[i,0]=conversion_co_train*S_mean_real[i,0]     
for i in range(0,S_mean_real.shape[0]):                     #convert PPFD to ETR
    S_mean_real[i,0]=a*(1-math.exp(-k*conversion_co_train*S_mean_real[i,0])) 

S = S_mean_real 
D_n = 3*(10**6)/ (3600/4)                                   #recommended DPI
x00 = np.zeros((1,len(S)))  
A = -np.ones((1,len(S)))  
b = sum(S) - D_n
lb = np.zeros((len(S),1))                                   #lower bound
lb[len(S)-int(samples_without_sunlight/2):len(S),0] = lb_etr 
lb[0:int(samples_without_sunlight/2),0] = lb_etr
ub = np.zeros((len(S),1))                                   #upper bound
for qq in range(0,len(S)):
    ub[qq][0] = min(U_LED,a-S[qq][0]-0.001)                 #0.001 works like an epsilon
steps=len(S)
f1 = 0
x = cp.Variable((steps,1)) 
for j in range(0,steps):                                    #cost function goes here
    f = C[0][j]*(math.log(a)-cp.log(a-S[j][0]-x[j][0]))     #for solving log use cp.log not np.log
    f1 = f1+ f 
objective = cp.Minimize(f1)
constraints = [lb <= x, x <= ub, A@x<=b]                    #All constraints should be in terms of matrix
prob = cp.Problem(objective, constraints)
prob.solve()                                                
pythonresult = x.value                                      #optimal light
for tt in pythonresult:
    if tt==None:
        pythonresult=ub
        break
pythonresult=np.where(pythonresult<=0,0,pythonresult)
X_Light[0:(int(samples_without_sunlight/2))] = pythonresult[0:(int(samples_without_sunlight/2))]  

#changing duty cycle of PWM for both methods before sunlight

D_n_step_time = (3*(10**6)/(3600/4)) / iterations            #for hueristic method

for ee in range(0,int(samples_without_sunlight/2)):  
    print('iteration:',ee,flush=True)
    p_p.ChangeDutyCycle((X_Light[ee,0]/U_LED)*100)           #prediction PWM changing
    print('duty_beforesunrise_p:',(X_Light[ee,0]/U_LED)*100,flush=True)
    X_Light1[ee][0] = min(D_n_step_time,U_LED)
    p_h.ChangeDutyCycle((X_Light1[ee][0]/U_LED)*100)         #heuristic PWM changing
    print('duty_beforesunrise_h:',(X_Light1[ee][0]/U_LED)*100,flush=True)
    if ee!=(int(samples_without_sunlight/2)-1):
    	time.sleep(900)                                      #change duty cycle at each time step

for i in range(0,(int(samples_without_sunlight/2))):           
    S1[i][0] = S[i][0]
Sunlight_Until_Now = np.zeros((iterations,1))

#############################################################################################
#############################################################################################
#Enable reading sunlight through ADS1115 

i2c = busio.I2C(board.SCL, board.SDA)                        # Create the I2C bus
ads = ADS.ADS1115(i2c)                                       # Create the ADC object using the I2C bus
chan = AnalogIn(ads, ADS.P0)                                 # Create single-ended input on channel 0
cal_factor = 100                                             # Sensor Calibration factor

#############################################################################################
#############################################################################################
# During the day

counter = 1
sample_in_each_zone = int(len(mean_daily_data)/zone_num)
M = np.ones((1,zone_num))*sample_in_each_zone
flag = 1                                                     # simulation begins from zone 
flag2 = 0                                                    # showing the switching from a zone to the next one
S_sensor=[]                                                  # sensor data (sunlight)

for i in range(int(samples_without_sunlight/2),iterations - (int(samples_without_sunlight/2)+1)):     
    
    # Reading from sensor (PPFD,inside the greenhouse)
    Sen=[]
    for aq in range (0,5):
        Sen.append(1000*chan.voltage*cal_factor)
        Sen.append(1000*chan.voltage*cal_factor)
        Sen.append(1000*chan.voltage*cal_factor)
        time.sleep(180)                                      #each 3 mins
    Sen=[0 if u < 0 else u for u in Sen]                     #due to sensor noise
    Sen=[500 if u > 500 else u for u in Sen]                 #set it based on month (500 for Dec)
    S_sensor_PPFD = statistics.mean(Sen)                     #calculate mean for the whole step time (15mins)
 
    S_sensor_wat = (S_sensor_PPFD/2.02)/0.4                  #convert to wat, outside of the greenhouse (the way training data was)
    print('iteration:',i,flush=True)
    print('S_sensor_PPFD:',S_sensor_PPFD,flush=True)
    S_Test[i,0]=a*(1-math.exp(-k*S_sensor_PPFD))             #sensor(ETR, inside)
    
    # Predicting sunlight
    Predicted_Day = []
    x0 = np.zeros((1,n*zone_num))
    snum=int(S_sensor_wat/(Z_Max_All[0,flag-1]/n))+1 
    if snum>12:
        snum=12
    statenum=snum+n*(flag-1)
    if flag2==0:
        x0[0,statenum-1]=1
    Predicted_Day1 = []
    flag1 = 0    
    for iii in range(flag-1,zone_num):   
        counterL = 2
        Predicted_Day11 = []
        if flag2==1:
            flag2=0
            for ij in range(0,n):
                if ((S_sensor_wat>=Z_All[flag-1][ij])  and  
                    (S_sensor_wat<Z_All[flag-1][ij+1])):
                    x0[0][ij+(flag-1)*n]=1       
        numSteps=M[0][iii]
        Predicted_Day11.append([float(np.dot(x0,Mean_Z.T))]) 
        for cc in range(0,int(numSteps)):
            x0 = np.matmul(x0,TM)
            Predicted_Day11.append([float(np.dot(x0,Mean_Z.T))])
            counterL+=1
        x0 = np.zeros((1,n*zone_num))
        if iii<zone_num-1:
            for ij in range(0,n):
                if ((Predicted_Day11[-1][0]>=Z_All[iii+1][ij])  and 
                (Predicted_Day11[-1][0]<Z_All[iii+1][ij+1])): #[-1][0] works as the last element of an array
                    x0[0][(ij+1)+((iii+1)*n)-1]=1
        if flag1>0:
            del Predicted_Day11[0]
        Predicted_Day11 = np.array((Predicted_Day11))
        if len(Predicted_Day1)==0:
            Predicted_Day1 = Predicted_Day11
        else:
            Predicted_Day1 = np.concatenate((Predicted_Day1,Predicted_Day11), axis=0) #axis 0 for writing in a column
        flag1+=1
    M[0][flag-1] = M[0][flag-1]-1
    if M[0][flag-1]==0:
        flag+=1
        flag2 = 1
    Predicted_Day = Predicted_Day1
   
    Predicted_Day[-1][0] = Mean_Z[0][((zone_num-1)*n)+2-1]
    S = np.concatenate((Predicted_Day,np.zeros((int(samples_without_sunlight/2),1))),axis=0)
    S[0,0] = S_sensor_wat
    for hh in range(0,S.shape[0]): 
        S[hh,0]=a*(1-math.exp(-k*conversion_co_train*S[hh,0]))    #convert to ETR, inside
    S1[i+1][0] = S[1][0]

    #############################################################################################
    #############################################################################################
    # Solving Optimization Problem
    
    D_n = (12/q)*3*(10**6)/(3600) - (float(sum(X_Light)) + float(sum(Sunlight_Until_Now)))
    x00 = np.ones((1,len(S)))
    A = -np.ones((1,len(S)))  
    b = sum(S) - D_n
    lb = np.zeros((len(S),1))
    lb[len(S)-int(samples_without_sunlight/2):len(S),0] = lb_etr
    ub = np.zeros((len(S),1))   
    for qq in range(0,len(S)):
        ub[qq][0] = min(U_LED,a-S[qq][0]-0.001) 
    steps = len(S) 
    C = C1[0,i-1:C1.shape[1]]
    C = C.reshape((1,-1))
    f1 = 0
    x = cp.Variable((steps,1)) 
    for j in range(0,steps):
        f = C[0][j]*(math.log(a)-cp.log(a-S[j][0]-x[j][0]))  
        f1 = f1+ f 
    objective = cp.Minimize(f1)
    constraints = [lb <= x, x <= ub, A@x<=b] 
    prob = cp.Problem(objective, constraints)
    prob.solve()  
    x = x.value 
    for yy in x:
        if yy==None:
            x=ub
            break
    x=np.where(x<=0,0,x)
    xxx[-1-len(x)+1:xxx.shape[0],counter-1] = x[:,0]    #Predicted needed supplied light from now on
    X_Light[i][0] = x[0][0]
    
    if  S_Test[i,0] < D_n_step_time:                    #calculate heuristic method supplemental light 
        X_Light1[i][0] = min(D_n_step_time-S_Test[i,0],U_LED)
    
    p_p.ChangeDutyCycle((X_Light[i][0]/U_LED)*100)       #prediction PWM changing
    print('duty_day_p:',(X_Light[i][0]/U_LED)*100,flush=True)
    p_h.ChangeDutyCycle((X_Light1[i][0]/U_LED)*100)      #heuristic PWM changing
    print('duty_day_h:',(X_Light1[i][0]/U_LED)*100,flush=True)
    Sunlight_Until_Now[i][0] = S_Test[i][0]
    counter+=1
    S_sensor.append(S_sensor_PPFD)                       #PPFD inside
    
    
#end of the for loop!
S1[i+2,0] = a*(1-math.exp(-k*Mean_Z[0][(zone_num-1)*n]))

#############################################################################################
#############################################################################################
# After sunset 

Sen=[]
for aq in range (0,5):                                    #read sunlight by sensor for the last time
    Sen.append(1000*chan.voltage*cal_factor)
    Sen.append(1000*chan.voltage*cal_factor)
    Sen.append(1000*chan.voltage*cal_factor)
    time.sleep(180)   
Sen=[0 if u < 0 else u for u in Sen]
Sen=[500 if u > 500 else u for u in Sen]
S_sunset = statistics.mean(Sen)
S_sensor.append(S_sunset)

S_sunset_ETR = a*(1-math.exp(-k*S_sunset))                #inside
D_n = 3*10**6/ (3600/(12/q)) - ( float(sum(X_Light)) + float(sum (Sunlight_Until_Now)) )
if S_sunset_ETR < D_n_step_time:
    X_Light1[i+1][0] = min(D_n_step_time-S_sunset_ETR,U_LED)
p_h.ChangeDutyCycle((X_Light1[i+1][0]/U_LED)*100)         #calculate heuristic method supplemental light for one step time
print("iteration:",i+1,flush=True)
print('duty_h:',(X_Light1[i+1][0]/U_LED)*100,flush=True)

S_sensor = np.array((S_sensor)).reshape((-1,1))
S_sensor = np.concatenate((nosunlight,S_sensor,nosunlight), axis=0)   # PPFD inside 

# Solving optimization Problem
S = []
for y in range(i+1,iterations):
    S.append(a*(1-math.exp(-k*S_sensor[y,0])))            #ETR inside
S = np.array((S)).reshape((-1,1))
x00 = np.ones((1,len(S)))
A = -np.ones((1,len(S)))  
b = sum(S) - D_n
lb = np.zeros((len(S),1))
lb[len(S)-int(samples_without_sunlight/2):len(S),0] = lb_etr
ub = np.zeros((len(S),1))   
for qq in range(0,len(S)):
    ub[qq][0] = min(U_LED,a-S[qq][0]-0.001) 
steps = len(S)
f1 = 0
x = cp.Variable((steps,1)) 
for j in range(0,steps):
    f = C[0][j]*(math.log(a)-cp.log(a-S[j][0]-x[j][0]))  
    f1 = f1+ f 
objective = cp.Minimize(f1)
constraints = [lb <= x, x <= ub, A@x<=b] 
prob = cp.Problem(objective, constraints)
prob.solve()  
x = x.value 
for zz in x:
    if zz==None:
        x=ub
        break
x=np.where(x<=0,0,x)
X_Light[i+1:,:]=x
print('x',x)
for ee in range(0,len(x)):
    print('iter_aftersunset:',ee,flush=True)
    p_p.ChangeDutyCycle((x[ee,0]/U_LED)*100)             #prediction PWM changing
    print('duty_aftersunset_p:',(x[ee,0]/U_LED)*100,flush=True)
    
    if ee!=0:
        X_Light1[ee+i+1][0] = min(D_n_step_time,U_LED)                       
        p_h.ChangeDutyCycle((X_Light1[ee+i+1][0]/U_LED)*100)                #heuristic PWM changing
        print('duty_aftersunset_h:',(X_Light1[ee+i+1][0]/U_LED)*100,flush=True)
    
    time.sleep(900)                                       #change duty cycle at each time step
    
p_p.ChangeDutyCycle(0)                                     #turning the lights off
p_h.ChangeDutyCycle(0)                                     #turning the lights off
Sunlight_Until_Now[i+1:,:] = S_Test[i+1:,:]

# Converting S_sensor to PPFD 
S_Test = S_sensor
S1_PPFD = np.zeros((len(S1),1))
for y in range(0,len(S1)):
    S1_PPFD[y,0] = 1/k*(math.log(a)-math.log(a-S1[y,0]))
S_Test_ETR = np.zeros((len(S_sensor),1))
for y in range(0,len(S_sensor)):
    S_Test_ETR[y,0]=a*(1-math.exp(-k*S_sensor[y,0]))         #ETR inside
S_Test = S_Test_ETR
S_Test_PPFD = np.zeros((len(S_Test_ETR),1))
for y in range(0,len(S_Test_ETR)):
    S_Test_PPFD[y,0] = 1/k*(math.log(a)-math.log(a-S_Test_ETR[y,0]))
    
#############################################################################################
#############################################################################################
# Solving Optimization Problem with Complete Information (Baseline)

S = S_Test                                              #ETR inside  
D_n = 3*(10**6)/ (3600/4)
x00 = np.zeros((1,len(S)))  
A = -np.ones((1,len(S)))  
b = sum(S) - D_n
lb = np.zeros((len(S),1))
lb[len(S)-int(samples_without_sunlight/2):len(S),0] = lb_etr
lb[0:int(samples_without_sunlight/2),0] = lb_etr
ub = np.zeros((len(S),1))   
C = C1
C = C.reshape((1,-1))
for qq in range(0,len(S)):
    ub[qq][0] = min(U_LED,a-S[qq][0]-0.001) 
steps=len(S)
f1 = 0
x = cp.Variable((steps,1)) 
for j in range(0,steps):
    f = C[0][j]*(math.log(a)-cp.log(a-S[j][0]-x[j][0]))  
    f1 = f1+ f 
objective = cp.Minimize(f1)
constraints = [lb <= x, x <= ub, A@x<=b] 
prob = cp.Problem(objective, constraints)
prob.solve()  
x_complete_information = x.value 
for ff in x_complete_information:
    if ff==None:
        x_complete_information=ub
        break
x_complete_information=np.where(x_complete_information<=0,0,x_complete_information)

#############################################################################################
#############################################################################################
# comparing cost of two methods and the baseline

f_min_global_cost = 0
for j in range(0,len(S_Test)):
    sum_global = C1[0][j]*((1/k)*(math.log(a)-math.log(a-S_Test[j][0]-x_complete_information[j][0]))-S_Test_PPFD[j][0])
    f_min_global_cost = f_min_global_cost + sum_global
    
f_min_prediction_cost = 0
for j in range(0,len(S_Test)):
    sum_prediction = C1[0][j]*((1/k)*(math.log(a)-math.log(a-S_Test[j][0]-X_Light[j][0]))-S_Test_PPFD[j][0])
    f_min_prediction_cost = f_min_prediction_cost + sum_prediction
    
error_cost_prediction = ((abs(f_min_global_cost-f_min_prediction_cost))/f_min_global_cost)*100
print('error_cost_prediction',error_cost_prediction,flush=True)

f_min_heuristic_cost = 0
for j in range(0,len(S_Test)):
    sum_heuristic = C1[0][j]*((1/k)*(math.log(a)-math.log(a-S_Test[j][0]-X_Light1[j][0]))-S_Test_PPFD[j][0])
    f_min_heuristic_cost = f_min_heuristic_cost + sum_heuristic
error_cost_Heuristic = ((abs(f_min_global_cost-f_min_heuristic_cost))/f_min_global_cost)*100
print('error_cost_Heuristic',error_cost_Heuristic,flush=True)

#############################################################################################
#############################################################################################
# Changing ETR to PPFD

S1_PPFD = np.zeros((len(S1),1))                                        #predicted sunlight
for tt in range(0,len(S1)):
    S1_PPFD[tt,0] = 1/k*(math.log(a)-math.log(a-S1[tt,0]))
    
x_complete_information_PPFD = np.zeros((len(x_complete_information),1))   #supplemental lighting provided by baseline
for y in range(0,len(x_complete_information)):
    x_complete_information_PPFD[y,0] = 1/k*(math.log(a)-math.log(a-x_complete_information[y,0]))
                    
X_Light1_PPFD = np.zeros((len(X_Light1),1))                               #supplemental lighting provided by heuristic
for y in range(0,len(X_Light1)):
    X_Light1_PPFD[y,0] = 1/k*(math.log(a)-math.log(a-X_Light1[y,0]))

X_Light_PPFD = np.zeros((len(X_Light),1))                                 #supplemental lighting provided by prediction
for y in range(0,len(X_Light)):
    X_Light_PPFD[y,0] = 1/k*(math.log(a)-math.log(a-X_Light[y,0]))
    
#############################################################################################
#############################################################################################
# Saving data on Rpi

np.savetxt('S1_PPFD_'+str(month)+'_'+str(day),S1_PPFD)
np.savetxt('S_Test_PPFD_'+str(month)+'_'+str(day),S_Test_PPFD)
np.savetxt('X_Light_PPFD_'+str(month)+'_'+str(day),X_Light_PPFD)
np.savetxt('X_Light1_PPFD_'+str(month)+'_'+str(day),X_Light1_PPFD)
np.savetxt('x_complete_information_PPFD_'+str(month)+'_'+str(day),x_complete_information_PPFD)
f_min_prediction_cost=np.array([[f_min_prediction_cost]])
f_min_global_cost=np.array([[f_min_global_cost]])
f_min_heuristic_cost=np.array([[f_min_heuristic_cost]])
np.savetxt('f_min_heuristic_cost_'+str(month)+'_'+str(day),f_min_heuristic_cost)
np.savetxt('f_min_global_cost_'+str(month)+'_'+str(day),f_min_global_cost)
np.savetxt('f_min_prediction_cost_'+str(month)+'_'+str(day),f_min_prediction_cost)
end = datetime.now()
print('End:',end)

#############################################################################################
#############################################################################################
