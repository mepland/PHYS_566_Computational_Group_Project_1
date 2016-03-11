import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

if(False):
	output_path = './output/dev'
	debugging = True





########################################################
########################################################
# Production Runs for paper 

if(False):
	top_output_path = './output/plots_for_paper'
	debugging = False

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


