import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import pickle as pickle
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick(add=1):
	_iter[0] += add

def set(value):
	_iter[0] = value + 1

def plot(name, value):
		_since_last_flush[name][_iter[0]] = value

def clear():
	_since_beginning.clear()
	_since_last_flush.clear()

def flush(path = "",title=None):
	prints = []

	for name, vals in list(_since_last_flush.items()):
		prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
		_since_beginning[name].update(vals)

		x_vals = np.sort(list(_since_beginning[name].keys()))
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)

		for i, j in zip(x_vals, y_vals):
			plt.annotate(str(j), xy=(i, j))

		plt.xlabel('iteration')
		plt.ylabel(name)

		if not title == None:
			plt.title(title)

		plt.savefig(os.path.join(path, name.replace(' ', '_')+'.png'))

	#print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
	_since_last_flush.clear()

	with open('log.pkl', 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
