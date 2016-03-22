import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
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

Dx = 3.0 # Delta x of our lattice 
Dy = 3.0 # Delta y of our lattice 
starting_Lx = 3*R_start # length in x of our starting lattice 
starting_Ly = 3*R_start # length in y of our starting lattice 

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
print 'Starting Lattice L x = %.1f' % starting_Lx
print 'Starting Lattice L y = %.1f' % starting_Ly

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
	placement_attempts = 0
	while overlap:
		placement_attempts += 1

		if placement_attempts > 10**4:
			print 'WARNING!! Unsuccessfully attempted to place a new walker on the starting circle 10**4 times, exiting!!'
			sys.exit()

		# generate random starting position on R_start circle
		theta = 2*np.pi*random.random()
		rand_x = R_start*math.cos(theta)
		rand_y = R_start*math.sin(theta)

		# Find x position on the lattice
		for i in range(int(starting_Lx/Dx)):
			if (-starting_Lx/2 + Dx*(i-0.5)) <= rand_x and rand_x < (-starting_Lx/2 + Dx*(i+0.5)):
				self.x = -starting_Lx/2 + Dx*i

		# Find y position on the lattice
		for i in range(int(starting_Ly/Dy)):
			if (-starting_Ly/2 + Dy*(i-0.5)) <= rand_y and rand_y < (-starting_Ly/2 + Dy*(i+0.5)):
				self.y = -starting_Ly/2 + Dy*i

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
	rMin = 2*max(starting_Lx, starting_Ly)
	for i in range(len(cluster)):
		r = math.sqrt( (self.x - cluster[i].x)**2 + (self.y - cluster[i].y)**2 )
		if r < rMin: rMin = r
	return rMin
    # end def for d_min()

    # Return spatial separation r from the seed/origin
    def r(self):
	return math.sqrt( self.x**2 + self.y**2 )
    # end def for r()

    # See if adjacent to an existing cluster point
    def touching(self, cluster = []):
	touch = False
	for i in range(len(cluster)):
		if cluster[i].x == self.x + Dx and cluster[i].y == self.y: touch = True # right
		elif cluster[i].x == self.x and cluster[i].y == self.y + Dy: touch = True # top
		elif cluster[i].x == self.x - Dx and cluster[i].y == self.y: touch = True # left
		elif cluster[i].x == self.x and cluster[i].y == self.y - Dy: touch = True # bottom

		elif cluster[i].x == self.x + Dx and cluster[i].y == self.y + Dy: touch = True # top right corner
		elif cluster[i].x == self.x - Dx and cluster[i].y == self.y + Dy: touch = True # top left corner
		elif cluster[i].x == self.x - Dx and cluster[i].y == self.y - Dy: touch = True # bottom left corner
		elif cluster[i].x == self.x + Dx and cluster[i].y == self.y - Dy: touch = True # bottom right corner

	return touch
    # end def for touching()

# end class for cluster_point

# Define a function to get a new cluster point
def new_cluster_point(index, cluster = []):
	if(debugging2): print 'Beginning new_cluster_point, index = %d ' % index	

	# repeatedly generate m_walker until one hits the cluster

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

# Define a function to generate cluster, fixed size if size > 1, or
# just grow until one point is on or beyond R_start if size <= 0
def gen_cluster(seed, size):
	random.seed(seed)

	# Set up the cluster with the seed at (0,0)
	cluster = []
	cluster.append(cluster_point(-1, cluster))
	cluster[0].x = 0.0
	cluster[0].y = 0.0
	cluster[0].fixed = True

	# If size > 0, make fixed size cluster
	if size > 0:
		for i in range(size-1):
			cluster.append(new_cluster_point(i, cluster))

		if(debugging): print 'cluster of size %d generated!!' % size

	elif size <= 0:
		i = 0
		while cluster[i].r() < R_start:
			i += 1
			cluster.append(new_cluster_point(i-1, cluster))

		if(debugging): print 'large cluster generated!!'

	return cluster
# end def for gen_cluster()

# Define a function to plot a cluster
def plot_cluster(fit_upper_cutoff, optional_title, m_path, fname, seed, cluster = []):
	if(debugging): print 'Beginning plot_cluster()'

	# Set up the figure and axes
 	fig = plt.figure('fig')
	ax = fig.add_subplot(111)
	ax.set_title(optional_title)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# adjust axis range
	ax.axis('scaled')
	axis_scale = 1.1
	ax.set_xlim((-axis_scale*R_start,axis_scale*R_start))
	ax.set_ylim((-axis_scale*R_start,axis_scale*R_start))

	# start list for legend entries/handles
 	legend_handles = []

	# plot the cluster, besides the seed
	for i in range(1, len(cluster)):
		cp = plt.Rectangle((cluster[i].x-Dx/2, cluster[i].y-Dy/2), Dx, Dy, color='blue', alpha=1, fill=True, label='Cluster')
		ax.add_artist(cp)

	# plot the seed, last so it's on top and you can see it separately in green
	cp_seed = plt.Rectangle((cluster[0].x-Dx/2, cluster[0].y-Dy/2), Dx, Dy, color='green', alpha=1, fill=True, label='Seed')
	ax.add_artist(cp_seed)

	legend_handles.append(cp)
	legend_handles.append(cp_seed)

	# make a circle on the starting position
	starting_radius_circle = plt.Circle((0,0), R_start, ls='dashed', color='grey', fill=False, label='$R_{\mathrm{Start}}$')
	ax.add_artist(starting_radius_circle)
	legend_handles.append(mlines.Line2D([], [], ls='dashed', color='grey', label='$R_{\mathrm{Start}}$'))

	# make a circle on the fit upper cutoff
	fit_upper_cutoff_circle = plt.Circle((0,0), fit_upper_cutoff, ls='dotted', color='darkmagenta', fill=False, label='$m(r)$ Fit Cut Off')
	ax.add_artist(fit_upper_cutoff_circle)
	legend_handles.append(mlines.Line2D([], [], ls='dotted', color='darkmagenta', label='$m(r)$ Fit Cut Off'))

	# draw legend
 	ax.legend(handles=legend_handles, bbox_to_anchor=(1.03, 1), borderaxespad=0, loc='upper left', fontsize='x-small')

	# Annotate
	ann_text = 'RNG Seed = %d\n$N =$ %d' % (seed, len(cluster))
	ann_text += '\n$\Delta x =$ %.1f, $\Delta y =$ %.1f' % (Dx, Dy)
	ann_text += '\n$R_{\mathrm{Start}} =$ %.1f' % (R_start)
	ann_text += '\n$d_{\mathrm{Kill}} =$ %.1f\n$t_{\mathrm{Kill}} =$ %G' % (d_kill, step_kill)

	ax.text(1.0415, 0.7, ann_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes)

	# Print it out
	make_path(m_path)
	# fig.savefig(m_path+'/'+fname+'.png', dpi=900)
	# if len(cluster) < 10**3: fig.savefig(m_path+'/'+fname+'.pdf')
	fig.savefig(m_path+'/'+fname+'.pdf')

	fig.clf() # Clear fig for reuse

	if(debugging): print 'plot_cluster() completed!!!'
# end def for plot_cluster()

# Define a function to plot a cluster's mass, and return it's fractal dimension
def plot_cluster_mass(fit_upper_cutoff, optional_title, m_path, fname, seed, cluster = []):
	if(debugging): print 'Beginning plot_cluster_mass()'

	nbins = 10
	counts = np.zeros(nbins)
	r = np.zeros(nbins)

	# Compute the mass/counts as a function of r for this cluster
	for i in range(len(cluster)):
		for j in range(1,nbins+1):
			if cluster[i].r() < j*(R_start/nbins): counts[j-1] += 1

	for i in range(1,nbins+1):
		r[i-1] = i*(R_start/nbins)

	# Create the ndarrays we can fit
	fit_counts = []
	fit_r = []

	for i in range(len(r)):
		if r[i] < fit_upper_cutoff:
			fit_counts.append(counts[i])
			fit_r.append(r[i])

	fit_counts_ndarray = np.array(fit_counts)
	fit_r_ndarray = np.array(fit_r)	

	# Set up the figure and axes
 	fig = plt.figure('fig')
	ax = fig.add_subplot(111)
	ax.set_title(optional_title)
	ax.set_xlabel('$r$')
	ax.set_ylabel('$m(r)$')


	# Save handles for legend
	legend_handles = []

	# Plot the data
	data_points = ax.scatter(r, counts, marker='o', label='$m(r)$', c='blue')
	legend_handles.append(data_points)

	# Make the plot log log
	ax.set_xscale('log')
	ax.set_yscale('log')

	# Fitting 
	########################################################

	########################################################
	# Define the fit function
	def fit_function(n_data, pow_fit, slope_fit):
		return slope_fit*pow(n_data, pow_fit)
	# end def fit_function

	# actually perform the fits
	# op_par = optimal parameters, covar_matrix has covariance but no errors on plot so it's incorrect...
	m_p0 = [1.65, 1.0]
	fit_status = True
	maxfev=m_maxfev = 2000

	try:
		op_par, covar_matrix = curve_fit(fit_function, fit_r_ndarray, fit_counts_ndarray, p0=m_p0, maxfev=m_maxfev)
	except RuntimeError:
		print sys.exc_info()[1]
		print 'curve_fit failed, continuing...'
		fit_status = False

	fit_text = 'Fit Function: $m(r) = b r^{d_{f}}$'

	# plot and annotate the fit
	if(fit_status):
		fit_line, = ax.plot(r, fit_function(r, *op_par), ls='solid', label='Fit', c='black')
		legend_handles.append(fit_line)

		fit_upper_cutoff_line = ax.axvline(x=fit_upper_cutoff, ls = 'dotted', label='Fit Cut Off', c='darkmagenta ')
		legend_handles.append(fit_upper_cutoff_line)

		fit_text += '\n$d_{f\,\mathrm{Fit}} =$ %.5f' % (op_par[0])
		fit_text += '\n$b_{\mathrm{Fit}} =$ %.5f' % (op_par[1])
	else:
		fit_text += '\nFit Failed'

	ax.text(0.025, 0.885, fit_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes)

	# Draw the expectation line
	expecation_line, = ax.plot(r, fit_function(r, *m_p0), ls='dashed', label='Expected', c='grey')
	legend_handles.append(expecation_line)


	'''
	# adjust axis range TODO
	x1_auto,x2_auto,y1_auto,y2_auto = ax.axis()
	ax.set_xlim(
	ax.set_ylim(
	'''

	# draw legend
 	ax.legend(handles=legend_handles, bbox_to_anchor=(0.98, 0.98), borderaxespad=0, loc='upper right', fontsize='x-small')

	# Annotate
	ann_text = 'RNG Seed = %d\n$N =$ %d' % (seed, len(cluster))
	ann_text += '\n$\Delta x =$ %.1f, $\Delta y =$ %.1f' % (Dx, Dy)
	ann_text += '\n$R_{\mathrm{Start}} =$ %.1f' % (R_start)
	ann_text += '\n$d_{\mathrm{Kill}} =$ %.1f\n$t_{\mathrm{Kill}} =$ %G' % (d_kill, step_kill)

	ax.text(0.79, 0.06, ann_text, bbox=dict(edgecolor='black', facecolor='white', fill=False), size='x-small', transform=ax.transAxes)

	# Print it out
	make_path(m_path)
	# fig.savefig(m_path+'/'+fname+'.png', dpi=900)
	# if len(cluster) < 10**3: fig.savefig(m_path+'/'+fname+'.pdf')
	fig.savefig(m_path+'/'+fname+'.pdf')

	fig.clf() # Clear fig for reuse

	if(debugging): print 'plot_cluster_mass() completed!!!'

	return op_par[0]

# end def for plot_cluster_mass()



# Define a wrapper function to do everything for part c
def part_c(fit_upper_cutoff, optional_title, m_path, fname, initial_seed):

	d_f_vals = [] # fractal dimension array

	for i in range(initial_seed,initial_seed+10):

		if (debugging): print '\n \t Seed #%d begins\n' % (i)

		cluster = gen_cluster(i, -1)
		plot_cluster(fit_upper_cutoff, optional_title, m_path, fname+'_seed_num_'+str(i), i, cluster)
		d_f_vals.append(plot_cluster_mass(fit_upper_cutoff, optional_title, m_path, fname+'_mass_seed_num_'+str(i), i, cluster))

	d_f_avr = np.sum(d_f_vals) / float(len(d_f_vals)) # average of d_f over all iterations
	d_f_std = np.std(np.asarray(d_f_vals)) # standard deviation of d_f over all iterations

	# print results of part c
	print '\n\n================= part c ================='
	print 'fractal dimensions array:'
	print d_f_vals
	print 'Averaged fractal dimension is %.2f' %d_f_avr 
	print 'standard deviation is %.2f ' %d_f_std
	print '=========================================='

# end def for part_c()


########################################################
########################################################
########################################################
# Finally, actually run things!

########################################################
########################################################
# Development Runs 

if(True):
	output_path = '../output/dev'
	debugging = True
	debugging2 = False	

	i = 7
	cluster = gen_cluster(i, -1)
	plot_cluster(75, '', output_path, 'test_seed_num_'+str(i), i, cluster)
	plot_cluster_mass(75, '', output_path, 'test_seed_num_'+str(i), i, cluster)


#	part_c(75, '', output_path, 'large_cluster', 7)



########################################################
########################################################
# Production Runs for paper 

if(False):
	top_output_path = '../output/plots_for_paper/problem_3'
	debugging = False
	debugging2 = False


	# TODO


########################################################
print '\n\nDone!\n'


