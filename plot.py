import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np 
from sklearn.metrics import confusion_matrix
import os

emotions = ["Marah", "Jijik", "Takut","Senang", "Sedih", "Terkejut", "Biasa saja"] 

import glob
def dir_plot(fn):
	paths = 'C:\\1.skripsi\\images\\plot'
	i = 0
	if any(filename.startswith(fn, 2) for filename in os.listdir(paths)):
		for j in range(len(glob.glob(paths+'*'+fn+'*'))):
			j+=1
		return paths + str(j) + '_' + fn + '_.png'
	else:
		return paths + str(i) + '_' + fn + '_.png'

def save_plot(fig, fn):
	paths = dir_plot(fn)
	fig.savefig(paths)

def plot_figure(x,y, title=None, save=False):
	fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7, _)) = plt.subplots(1, 8, figsize=(17, 11)) 
	_.set_visible(False)

	for index, label in enumerate(emotions):
		xj = []
		for j in range(len(y)):
			if y[j] == index:
				xj.append(x[j])
			else:
				j+1
		img = np.array(xj[0], dtype=np.uint8).reshape(48, 48)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		exec(f"ax{index + 1}.imshow(img)")
		exec(f"ax{index + 1}.set_title(label.title())")
	if title is not None and save is True:
		fn = 'figures_of_' + title
		save_plot(fig, fn)
	fig.tight_layout()

def cm_plot(y_true, y_pred,title=None, save=False):

	cf = confusion_matrix(y_true, y_pred)
	plt.subplots(figsize=(16,9))
	ax = sns.heatmap(cf, annot=True, cmap='Blues', linewidths=0.5)
	ax.set_title('Confusion Matrix ' + title + '\n\n');
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ');
	ax.xaxis.set_ticklabels(emotions)
	ax.yaxis.set_ticklabels(emotions)
	if title is not None and save is True:
		fn = 'confusion_matrix_'+ title
		save_plot(plt, fn)
	plt.show()

def history_plot(history, title=None, save =False):
	figure, axis = plt.subplots(figsize=(10,7))
	axis.plot(history.history['accuracy'])
	axis.plot(history.history['loss'])
	axis.set_title('Accuracy and Loss ' + title)
	axis.legend(['accuracy','loss'], loc='upper left')
	axis.set(xlabel = 'epoch',ylabel='accuracy')
	if title is not None and save is True:
		fn = 'graph_bpn_' + title
		save_plot(figure, fn)
	plt.show()

def plot_data_labels(labels):
	e, c_e = np.unique(labels, return_counts=True)
	plt.figure(figsize=(10,6))
	plt.bar(np.arange(len(e)), c_e)
	plt.xticks(np.arange(len(e)), e)
	plt.xlabel("Emosi")
	plt.ylabel("Banyak Gambar")
	plt.show()