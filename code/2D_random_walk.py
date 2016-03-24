###############2D Random Walk Test2

############################Preliminaries######################################
import numpy as np                        #Import Needed Libraries
import matplotlib.pyplot as plt
import random
from math import *
import os


def make_path(path):
    """
    Create the output diretory.
    :param path: path of the directory.
    :return: null.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % output_path)



#################################Part A########################################

n = 100                                      #Number of Steps
m = 10000.0                                 #Number of Random Walks

x2ave=np.asarray([0.0]*n)                  #Array for average x position squared
y2ave=np.copy(x2ave)                       #Array for average y position squared
xAve=np.copy(x2ave)                        #Array for average x position squared 
dist2=np.copy(x2ave)                        #Array for average distance squared


for j in range(int(m)):
    x = 0                          #Initialize x position
    y = 0                          #Initialize y position
    for i in range(n):  
        r = random.random()
        if r <= 0.25:
            x += 1
        elif r <= 0.5:
            x -= 1
        elif r <= 0.75:
            y += 1
        else:
            y -= 1
        xAve[i] += x                            #Find the Averages Over the Number of Random Walks
        x2ave[i] += x ** 2
        dist2[i] += x ** 2 + y ** 2

xAve /= m
x2ave /= m
dist2 /= m
    
distNew = dist2[3:100]                        #Get the dist for n>3 up to n=100
xAveNew = xAve[3:100]                        #Get the xAve for n>3 up to n=100
x2aveNew = x2ave[3:100]                      #Get the x2ave for n>3 up to n=100

#################################Part B########################################

steps=np.arange(4, n + 1, 1)                  #Get an array of steps to use for plotting



coefficients=np.polyfit(steps, distNew, 1)   #Fit the Data Using a Linear Polyfit
slope=coefficients[0]                      #Get the Slope Coefficient
print 'D = %f' % (slope / 4.0)             # Print Diffusion constant
diffCoeff=slope/2                          #Get the Diffusion Coefficient

eq=np.poly1d(coefficients)           #Get the Equation of the Line with a symbolic independent variable
eqSteps=eq(steps)                    #Plug in steps for independent variable in eq


###############################Plotting########################################   
plots_path = '../output/plots_for_paper/problem_1'
# make path
make_path(plots_path)
###############################Part A-xn####################################### 
plt.figure()  
plt.plot(steps, xAveNew,'r.', label="Mean Distance")
#plt.title(r'$\langle x_{n} \rangle$' + ' Plot', fontsize=15)   #Plot Annotations
plt.xlabel('Time', fontsize=12)
plt.ylabel('Distance Travelled', fontsize=12)
#plt.legend()
plt.savefig(plots_path + '/xn_Plot.pdf')               #Save the Plot as a PDF
plt.close()

##############################Part A-xn^2###################################### 
plt.figure()  
plt.plot(steps, x2aveNew,'r.', label="Mean Square Distance")
#plt.title(r'$\langle x_{n}^{2} \rangle$' + ' Plot', fontsize=15)   #Plot Annotations
plt.xlabel('Time', fontsize=12)
plt.ylabel('Distance Travelled Squared', fontsize=12)
#plt.legend()
plt.savefig(plots_path + '/xn2_Plot.pdf')               #Save the Plot as a PDF
plt.close()

###########################Part B-Diffusive Motion#############################  
plt.figure()  
plt.plot(steps, distNew,'r.', label="Mean Square Distance")
plt.plot(steps, eqSteps,'b',linewidth=2, label="Linear Fit")
#plt.title("Diffusive Motion Plot", fontsize=15)   #Plot Annotations
plt.xlabel('Time', fontsize=12)
plt.ylabel('Distance Travelled Squared', fontsize=12)
plt.legend(loc='upper left')
plt.savefig(plots_path + '/Diffusion_Plot.pdf')               #Save the Plot as a PDF
plt.close()

