# A helper module for various sub-tasks
from time import time
import numpy as np
import random
import contextlib
import os, io
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print(f'Time taken by {func.__name__} is {end-start:.4f} seconds')
		return val
	return new_func

def delta(x, x_0):
	"""
	Description:
		Dirac delta function

	Args:
		x: input
		x_0: point where the mass is located

	Returns:
	 	eiter 0.0 or 1.0
	"""
	return 1.0 if np.array_equal(x, x_0) else 0.0

class Picker(object):
    """
    A class defining an object-picker from an array
    """
    def __init__(self, array):
        """
        array = array of objects to pick from
        """
        self.array = array

    def equidistant(self, objs_to_pick, start_pt = 0):
        """
		Description:
        	Picks objs_to_pick equidistant objects starting at the location start_pt
        Returns:
			the picked objects
        """
        increment = int((len(self.array) - start_pt)/objs_to_pick)
        if increment < 1:
            return self.array
        else:
            new_array = [0]*objs_to_pick
            j = start_pt
            for i in range(objs_to_pick):
                new_array[i] = self.array[j]
                j += increment
        return np.array(new_array)

def normalize_small(numbers, threshold = 50):
	log_numbers = [np.log(number) for number in numbers]
	max_log = np.max(log_numbers)
	for i, number in enumerate(numbers):
		if max_log - log_numbers[i] > threshold:
			number[i] = 0.0

def KL_div_MC(p, q, samples):
	result = 0.0
	for x in samples:
		px = p(x)
		result += px*np.log(px/q(x))
	return result/len(samples)

def TV_dist_MC(p, q, samples):
	result = 0.0
	for x in samples:
		result +=np.abs(p(x)-q(x))
	return 0.5*result/len(samples)

def TV_dist_MC_avg(p, q, samples, batch):
	dist = 0.0
	for i in range(int(len(samples)/batch)):
		result = 0.0
		for x in samples[i*batch: (i+1)*batch]:
			result +=np.abs(p(x)-q(x))
		dist += result/batch
	return 0.5*dist/(len(samples)/batch)


@contextlib.contextmanager
def silencer():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout 


def grid_scatter_plotter(data_list, label_list, color_list, pairs, n_rows, n_cols, fig_name,\
						 xlabel_size, ylabel_size, title_size, tick_size,\
						 wspace, hspace, s=10, figsize=4, size_list=None):

	fig = plt.figure(figsize=(n_cols*figsize, n_rows*figsize))
	axes = [fig.add_subplot(n_rows, n_cols, j+1) for j in range(len(pairs))]
	
	if size_list is None:
		size_list = [s*np.ones(len(data_list[0]))] * len(axes)
	else:
		for i, size in enumerate(size_list):
			size = np.array(size)
			size = (size - size.mean()) / size.std()
			size -= 2.*min(size) 
			size_list[i] = s*size
	# print(size_list[0])
	for i, ax in enumerate(axes):
		j, k = pairs[i]
		ax.scatter(data_list[j], data_list[k], s=size_list[i], c=color_list[i], cmap='viridis')
		# ax.set_xlabel(label_list[j], fontsize=xlabel_size)
		ax.set_ylabel(label_list[k], fontsize=ylabel_size)
		ax.set_title('{} vs {}'.format(label_list[j], label_list[k]), fontsize=title_size)
	fig.subplots_adjust(wspace=wspace, hspace=hspace)
	plt.savefig(fig_name)
	plt.show()


def grid_heat_plotter(x, y, data_list, xlabel, ylabel, label_list, n_rows, n_cols, fig_name,\
						xlabel_size, ylabel_size, title_size, tick_size,\
						wspace, hspace, figsize=4, show=True):

	fig = plt.figure(figsize=(n_cols*figsize, n_rows*figsize))
	axes = [fig.add_subplot(n_rows, n_cols, j+1) for j in range(len(data_list))]
	
	for i, ax in enumerate(axes):
		im = ax.pcolormesh(x, y, data_list[i].T, cmap='viridis')
		ax.set_title('{}'.format(label_list[i]), fontsize=title_size)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical')
	
	fig.subplots_adjust(wspace=wspace, hspace=hspace)
	fig.supylabel(ylabel, fontsize=ylabel_size)
	fig.supxlabel(xlabel, fontsize=xlabel_size)
	plt.savefig(fig_name)
	if show:
		plt.show()
	else:
		plt.close()

class ExperimentLogger:
    def __init__(self, folder, description):
        self.folder = folder
        self.description = description
        self.objects = []
        self.logFile = folder + '/experiment_log.txt'
        with open(self.logFile, 'w') as file:
            file.write("=======================================================================\n")
            file.write("This is a short description of this experiment\n")
            file.write("=======================================================================\n")
            descriptionWithNewLine= description.replace('. ', '.\n')
            file.write(f"{descriptionWithNewLine}\n\n\n")
        
    def addSingle(self, object):
        self.objects.append(object)
        with open(self.logFile, 'a') as file:
            file.write("=======================================================================\n")
            file.write(f"Object: {object.name}\n")
            file.write("=======================================================================\n")
            for key, value in object.__dict__.items():
                file.write(f"{key} = {value}\n")
                file.write("-----------------------------------------------------------------------\n")
            file.write("\n\n")
    
    def add(self, *objects):
        for object in objects:
            self.addSingle(object)

def makeDir(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def figStart(figsize):
	fig = plt.figure(figsize=figsize)
	return fig, fig.add_subplot(111)



def file2str(folder_path, search_string, file_extensions=None, encoding='utf-8'):
    """
    Search for files under folder_path that contain search_string.

    Args:
        folder_path (str): Root directory to search in.
        search_string (str): Substring to look for.
        file_extensions (list of str, optional): If provided, only files
            ending with these extensions (e.g. ['.txt', '.py']) are searched.
            Case‑insensitive. Defaults to None (all files).
        encoding (str, optional): File encoding to use when reading.
            Defaults to 'utf-8'.

    Returns:
        List[str]: Paths to files where search_string was found.
    """
    matches = []
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            # If filtering by extension, skip others
            if file_extensions is not None:
                if not any(fname.lower().endswith(ext.lower()) 
                           for ext in file_extensions):
                    continue

            path = os.path.join(root, fname)
            try:
                with open(path, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        if search_string in line:
                            matches.append(path)
                            break  # stop reading this file
            except Exception as e:
                # Could log or print a warning here
                print(f"[Warning] Could not read {path}: {e}")
    return matches
