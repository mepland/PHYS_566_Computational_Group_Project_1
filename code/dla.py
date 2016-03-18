import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import math
import random

########################################################
# Define a function to create the output dir
# If it already exists don't crash, otherwise raise an exception
# Adapted from A-B-B's response to http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
# Note in python 3.4+ 'os.makedirs(output_path, exist_ok=True)' would handle all of this...
def make_path(path):
	try: 
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % output_path)
# end def for make_path


########################################################
# Set fixed/global parameters

R_start = 100.0 # random walker starting point radius

Dx = 1.0 # Delta x of our lattice 
Dy = 1.0 # Delta y of our lattice 
Lx = 4*R_start/Dx # length in x of our lattice 
Ly = 4*R_start/Dy # length in y of our lattice 

d_kill = 120.0 # random walker kill distance
step_kill = 10000 # random walker kill step


########################################################
# Print out fixed values
print '\nBeginning dla.py'
print '\nFixed Parameters are:'
print '---------------------------------------------'

print '\nRandom Walker Starting Radius R_start = %.1f ' % R_start

print '\nLattice Delta x = %.1f' % Dx
print 'Lattice Delta y = %.1f' % Dy
print 'Lattice L x = %.1f' % Lx
print 'Lattice L y = %.1f' % Ly

print '\nRandom Walker Kill Distance, d_kill = %.1f' % d_kill
print 'Random Walker Kill Step, step_kill = %d' % step_kill


print '\n---------------------------------------------'
print '---------------------------------------------\n'

########################################################
########################################################

########################################################
# Define cluster_point class to hold all the relevant parameters of a cluster point
# We will also use it as our random walker
class cluster_point:

    # constructor
    def __init__(self, index, cluster = []):
	self.index = index
	self.step_number = 0
	self.fixed = False

	overlap = True
	while overlap:
		# generate random starting position on R_start circle
		theta = 2*np.pi*random.random()
		rand_x = R_start*math.cos(theta)
		rand_y = R_start*math.sin(theta)

		# Find x position on the lattice
		for i in range(int(Lx/Dx)):
			if (-Lx/2 + Dx*(i-0.5)) <= rand_x and rand_x < (-Lx/2 + Dx*(i+0.5)):
				self.x = -Lx/2 + Dx*i

		# Find y position on the lattice
		for i in range(int(Ly/Dy)):
			if (-Ly/2 + Dy*(i-0.5)) <= rand_y and rand_y < (-Ly/2 + Dy*(i+0.5)):
				self.y = -Ly/2 + Dy*i

		# make sure we don't overlap an existing cluster point, repeat from the top if we do
		# if len(cluster) == 0 this is the seed and we don't need to check...
		overlap = False
		j = 0
		while j < len(cluster) and not overlap:
			if cluster[j].x == self.x and cluster[j].y == self.y: overlap = True
			j += 1

    # end def for constructor

    # Execute one random walk step
    def random_walk_step(self):
	# make sure we don't try to move a fixed cluster point
	if self.fixed:
		print 'ERROR!! Attempted to walk a fixed cluster point, exiting!!'
		sys.exit()

	# Perform the random walk 
	self.x += Dx*(-1 + 2*round(random.random()) ) # move left or right in x
	self.y += Dy*(-1 + 2*round(random.random()) ) # move up or down in y

	# increment the step number
	self.step_number += 1

	return None # return None explicitly, python will automatically but this is clearer to the reader
    
    # end def for random_walk_step()

    # Return spatial separation d from the nearest cluster point
    def d_min(self, cluster = []):
	rMin = 2*max(Lx, Ly)
	for i in range(len(cluster)):
		r = math.sqrt( (self.x - cluster[i].x)**2 + (self.y - cluster[i].y)**2 )
		if r < rMin: rMin = r
	return rMin
    # end def for d_min()

    # See if adjacent to an existing cluster point
    def touching(self, cluster = []):
	touch = False
	for i in range(len(cluster)):
		if cluster[i].x == self.x + Dx and cluster[i].y == self.y: touch = True # right
		elif cluster[i].x == self.x and cluster[i].y == self.y + Dy: touch = True # top
		elif cluster[i].x == self.x - Dx and cluster[i].y == self.y: touch = True # left
		elif cluster[i].x == self.x and cluster[i].y == self.y - Dy: touch = True # bottom

	return touch
    # end def for touching()

# end class for cluster_point

# Define a function to get a new cluster point
def new_cluster_point(index, cluster = []):
	if(debugging): print 'Beginning new_cluster_point, index = %d ' % index	

	# repeatly generate m_walker until one hits the cluster

	status = 0
	while status != 1:
	
		status = 0
		m_walker = cluster_point(index, cluster)
	
		# walking loop until one of the three halt conditions is met, ie status = 1, 2, or 3
		while status == 0:
		
			# Check to see if the walker hit a cluster point, and can join the cluster
			if(m_walker.touching(cluster)):
				status = 1
				m_walker.fixed = True
				if(debugging2): print 'Hit the Cluster!!'
				continue
	
			# Check to see if the walker is too far away 
			if(m_walker.d_min(cluster) > d_kill):
				status = 2
				if(debugging2): print 'Walked too far, d_min = %.2f > d_kill = %.2f on step_number = %d ' % (m_walker.d_min(cluster), d_kill, m_walker.step_number)
	
			# Check to see if the walker has run too long 
			if(m_walker.step_number > step_kill):
				status = 3
				if(debugging2): print 'Walked too long, d_min = %f ' % m_walker.d_min(cluster)
	
			# Perform another random walk step 
			m_walker.random_walk_step()
	
	# return the cluster point
	return m_walker
# end def for new_cluster_point()



# Define a function to plot and fit the data
def plot(optional_title, m_path, fname):
	if(debugging): print 'Beginning plot() for fname: '+fname	

	# create the ndarrays to plot
	data_ndarray = np.array()#TODO

	# Set up the figure and axes
        fig = plt.figure('fig')
        ax = fig.add_subplot(111)
        ax.set_title(dist+' $x$'+optional_title)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

	# plot the graph/histogram TODO

	'''
	# Fitting TODO
	########################################################

	########################################################
	# Define the linear fit function
	def linear_fit_function(n_data, offset_fit, slope_fit):
	        return offset_fit + slope_fit*n_data
	# end def linear_fit_function

	# actually perform the fits
	# op_par = optimal parameters, covar_matrix has covariance but no errors on plot so it's incorrect...

	linear_p0 = [1.0, 0.0]
	linear_fit_status = True

	maxfev=m_maxfev = 2000

	fit_text = ''

	try:
		linear_op_par, linear_covar_matrix = curve_fit(linear_fit_function, bins, n, p0=linear_p0, maxfev=m_maxfev)
	except RuntimeError:
		print sys.exc_info()[1]
		print 'linear curve_fit failed, continuing...'
		linear_fit_status = False 
	
	# plot the fit
	if(linear_fit_status):
		linear_fit_line, = ax.plot(bins, linear_fit_function(bins, *linear_op_par), ls='dashed', label='Linear Fit', c="black")
	
	# Write out the fit parameters
	fit_text = 'Linear Fit Function: Pr$(x) = a + b x$' 
	if(linear_fit_status):
		fit_text += '\n$a_{\mathrm{Expected}} =$ %2.2f, $a_{\mathrm{Fit}} =$ %2.5f' % (linear_p0[0], linear_op_par[0])
		fit_text += '\n$b_{\mathrm{Expected}} =$ %2.2f, $b_{\mathrm{Fit}} =$ %2.5f' % (linear_p0[1], linear_op_par[1])
	else:
		fit_text += '\nLinear Fit Failed'

	# Print the fit parameters
	ax.text(0.025, 1-0.03, fit_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes, va='top')
	'''

	'''
	# adjust axis range
	x1,x2,y1,y2 = ax.axis()
	ax.set_ylim((y1,1.20*y2))
	
	# Draw the legend
	ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0, fontsize='x-small')
	'''

	# Annotate
	ann_text = 'TODO'

	ax.text(0.77, 0.88, ann_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes, va='top')

	# Print it out
	make_path(m_path)
	fig.savefig(m_path+'/'+fname+'.pdf')	

	fig.clf() # Clear fig for reuse

	if(debugging): print 'plot() completed!!!'
# end def for plot()



########################################################
########################################################
########################################################
# Finally, actually run things!

########################################################
########################################################
# Development Runs 

if(True):
	output_path = '../output/problem_3/dev'
	debugging = True
	debugging2 = True

	# Set up the cluster with the seed at (0,0)
	cluster = []
	cluster.append(cluster_point(-1, cluster))
	cluster[0].x = 0.0
	cluster[0].y = 0.0
	cluster[0].fixed = True


	for i in range(5):
		cluster.append(new_cluster_point(i, cluster))


########################################################
########################################################
# Production Runs for paper 

if(False):
	top_output_path = '../output/plots_for_paper/problem_3'
	debugging = False
	debugging2 = False

        # Part a
        ########################################################
        print '\nPart a:'
        output_path = top_output_path+'/part_a'

	# TODO

	# Part b
        ########################################################
        print '\nPart b:'
        output_path = top_output_path+'/part_b'

	# TODO



########################################################
print '\n\nDone!\n'


