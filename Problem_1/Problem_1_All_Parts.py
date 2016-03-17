###############2D Random Walk Test2

############################Preliminaries######################################
import numpy as np                        #Import Needed Libraries
import matplotlib.pyplot as plt
import random

#################################Part A########################################

n=101                                      #Number of Steps
m=int(10**4)                                 #Number of Random Walks

x2ave=np.asarray([0.0]*n)                  #Array for average x position squared
y2ave=np.copy(x2ave)                       #Array for average y position squared
xAve=np.copy(x2ave)                        #Array for average x position squared 
dist=np.copy(x2ave)                        #Array for average distance squared


for j in range(m):
    random.seed()                #Initialize By Setting Seed Value
    x=0                          #Initialize x position
    y=0                          #Initialize y position
    for i in range(n):  
        r=random.random()
        if r<0.5:                       #Outer if statement determines whether motion is in x or y direction   
            rx=random.random()
            if rx<0.5:
                x=x+1
            else:                       #Inner if statement determines whether motion is in positive or negative direction
                x=x-1
            x2ave[i]=x2ave[i]+(x**2)
            xAve[i]=xAve[i]+x
        else:
            ry=random.random()
            if ry<0.5:
                y=y+1
            else:
                y=y-1
            y2ave[i]=y2ave[i]+(y**2)
    xAve[i]=xAve[i]/m                            #Find the Averages Over the Number of Random Walks
    x2ave[i]=x2ave[i]/m
    y2ave[i]=y2ave[i]/m

for i in range(n):                               #Find the dist from the start by adding x and y 
    dist[i]=x2ave[i]+y2ave[i] 
    
distNew=dist[3:99]                        #Get the dist for n>3 up to n=100
xAveNew=xAve[3:99]                        #Get the xAve for n>3 up to n=100
x2aveNew=x2ave[3:99]                      #Get the x2ave for n>3 up to n=100

#################################Part B########################################

steps=np.arange(4,i,1)                  #Get an array of steps to use for plotting



coefficients=np.polyfit(steps, distNew, 1)   #Fit the Data Using a Linear Polyfit
slope=coefficients[0]                      #Get the Slope Coefficient
diffCoeff=slope/2                          #Get the Diffusion Coefficient

eq=np.poly1d(coefficients)           #Get the Equation of the Line with a symbolic independent variable
eqSteps=eq(steps)                    #Plug in steps for independent variable in eq


###############################Plotting########################################   

###############################Part A-xn####################################### 
plt.figure()  
plt.plot(steps, xAveNew,'r.',Label="Mean Distance")
plt.title("xn Average Plot", fontsize=15)   #Plot Annotations
plt.xlabel('Time', fontsize=12)
plt.ylabel('Distance Travelled', fontsize=12)
plt.legend()
plt.savefig("xn_Plot.pdf")               #Save the Plot as a PDF

##############################Part A-xn^2###################################### 
plt.figure()  
plt.plot(steps, x2aveNew,'r.',Label="Mean Square Distance")
plt.title("xn^2 Average Plot", fontsize=15)   #Plot Annotations
plt.xlabel('Time', fontsize=12)
plt.ylabel('Distance Travelled Squared', fontsize=12)
plt.legend()
plt.savefig("xn2_Plot.pdf")               #Save the Plot as a PDF

###########################Part B-Diffusive Motion#############################  
plt.figure()  
plt.plot(steps, distNew,'r.',Label="Mean Square Distance")
plt.plot(steps, eqSteps,'b',linewidth=2, Label="Linear Fit")
plt.title("Diffusive Motion Plot", fontsize=15)   #Plot Annotations
plt.xlabel('Time', fontsize=12)
plt.ylabel('Distance Travelled Squared', fontsize=12)
plt.legend()
plt.savefig("Diffusion_Plot.pdf")               #Save the Plot as a PDF

