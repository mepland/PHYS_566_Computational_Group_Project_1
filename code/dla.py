import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
	if(debugging): print 'Beginning new_cluster_point, index = %d ' % index	

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

# Define a function to generate cluster
def gen_cluster(seed, size):
	random.seed(seed)

	# Set up the cluster with the seed at (0,0)
	cluster = []
	cluster.append(cluster_point(-1, cluster))
	cluster[0].x = 0.0
	cluster[0].y = 0.0
	cluster[0].fixed = True

	# Fill the rest of the points
	for i in range(size-1):
		cluster.append(new_cluster_point(i, cluster))

	return cluster
# end def for gen_cluster()


# Define a function to plot a cluster
def plot_cluster(optional_title, m_path, fname, seed, cluster = []):
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

	# draw legend
 	ax.legend(handles=legend_handles, bbox_to_anchor=(1.03, 1), borderaxespad=0, loc='upper left', fontsize='x-small')

	# Annotate
	ann_text = 'Seed = %d\n$N =$ %.2G' % (seed, len(cluster))
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

# Define a wrapper function to generate and plot a cluster
def gen_and_plot_cluster(optional_title, m_path, fname, seed, size):
	cluster = gen_cluster(seed, size)
	plot_cluster(optional_title, m_path, fname, seed, cluster)
# end def for gen_and_plot_cluster()


# Define a function to generate a R_start sized cluster and plot it
def gen_large_cluster(optional_title, m_path, fname, seed):
	if(debugging): print 'Beginning gen_large_cluster()'

	random.seed(seed)

	# Set up the cluster with the seed at (0,0)
	cluster = []
	cluster.append(cluster_point(-1, cluster))
	cluster[0].x = 0.0
	cluster[0].y = 0.0
	cluster[0].fixed = True

	# keep adding points until one is 'near' R_start
	# r_large = R_start - math.sqrt( Dx**2 + Dy**2 )
	# if(debugging): print 'r_large = %f' % r_large

	# keep adding points until one is on or beyond R_start
	i = 0
	while cluster[i].r() < R_start:
		i += 1
		cluster.append(new_cluster_point(i-1, cluster))
	if(debugging): print 'large cluster generated!!'

	plot_cluster(optional_title, m_path, fname, seed, cluster)

	if(debugging): print 'gen_large_cluster() completed!!!'	
# end def for gen_large_cluster()


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


	# cluster1 = gen_cluster(7, 5)
	# plot_cluster(' cluster1', output_path, 'cluster1', 7, cluster1)

	# gen_and_plot_cluster(optional_title, m_path, fname, seed, size)
#	gen_and_plot_cluster('', output_path, 'test1', 7, 10)

	# gen_large_cluster(optional_title, m_path, fname, seed)
	gen_large_cluster('', output_path, 'test_large', 7)



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


