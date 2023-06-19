from random import sample
import numpy as np
import pandas as pd
import cv2
import os
from os.path import abspath, join, dirname
from inspect import getsourcefile

from PIL import Image as im

# Find Faces
def classifiers(name:str) -> cv2.CascadeClassifier:
    retval = None
    cv2_path = os.path.dirname(cv2.__file__)
    data_path = os.path.join(cv2_path, "data")

    for f in os.listdir(data_path):
        fname,ext = os.path.splitext(f)
        if ext == '.xml' and fname == name:
            data_file = os.path.join(data_path, f).replace("\\", "/")
            retval = cv2.CascadeClassifier(data_file)
    
    return retval

def find_faces(images, name:str, crop=False):
	'''
	image : object
		
	object image dari openCV

	name : str
		nama file untuk CascadeClassifier terutama deteksi wajah contoh:
			
			"haarcascade_frontalface_alt"
			
			"haarcascade_frontalface_alt_tree"
			
			"haarcascade_frontalface_default"
	'''
	clf = classifiers(name)
	if crop is True:
		images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
		coordinates = locate_faces(images, clf)
		cropped_faces = [images[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
		normalized_faces = [normalized_face(face) for face in cropped_faces]
		# normalized_faces = normalized_face(cropped_faces)
		return zip(normalized_faces, coordinates)
	else :
		coordinates = locate_faces(images, clf)
		return coordinates

def normalized_face(face):
	# face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	face = cv2.resize(face, (48, 48))
	return face

def locate_faces(images, clf):
	faces = clf.detectMultiScale(
		images,
		scaleFactor = 1.1,
		minNeighbors =6,
		minSize = (30,30)
	)
	return faces

# Read images
def read_images(categories, path):
	'''
	membaca gambar dari sebuah folder

	categories : subfolder

	path : folder

	return :
		[X, y]
	'''
	path = dirname(abspath(getsourcefile(lambda:0)))
	X, y = []
	for i, category in enumerate(categories):
		samples = os.listdir(join(path, category))
		samples = [s for s in samples if 'png' or 'jpg' or 'jpeg' in s]
		for sample in samples:
			img = cv2.imread(join(path, category, sample), cv2.IMREAD_GRAYSCALE)
			# img = cv2.resize(img, (48,48))
			# img = img.reshape([1, 48*48])
			X.append(img)
			y.append(i)
	retval = [X, y]

	return retval

# Read CSV
def read_csv(path):
	'''
	Membaca file CSV

	path = folder + filename

	return :
		images, labels, usage

		or 

		images, labels
	'''
	df = pd.read_csv(path)
	pixels = df.loc[:, 'pixels'].values
	labels = df.loc[:, 'emotion'].values
	images = []
	try :
		for pixel in pixels:
			pixel = [int(t) for t in pixel.split(',')]
			image = im.fromarray(np.array(pixel).reshape(48,48))
			images.append(image)
	except ValueError:
		for pixel in pixels:
			pixel = [int(t) for t in pixel.split(' ')]
			image = im.fromarray(np.array(pixel).reshape(48,48))
			images.append(image)		
	try : 
		usage = df.loc[:, 'Usage'].values
	except KeyError:
		return images, labels
	return images, labels, usage

# Normalize 
def normalize(X, low, high, dtype=None):
	'''
	normalisai data
	'''
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	X = X - float(minX)
	X = X / float((maxX - minX))

	X = X * (high - low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

# As Row Matrix
def asRowMatrix(X):
	'''
	membuat baris matrix dari multi dimensi data menjadi 1 dimensi data

	'''
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0,X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat

# # fungsi equalizer preprosessing
def image_equalizer(images):
	'''
	image euqalizer preprosessing

	'''
	new_images = []
	for i in range(len(images)):
		histo = np.histogram(images[i], 256, (0, 255))[0]
		cdf = histo.cumsum()
		cdf_eq = (256-1)*(cdf/float(2304))
		cq = cdf_eq[images[i]]
		new_images.append(np.asarray(cq, dtype=np.uint8).reshape(1, 2304))
	return new_images

def video_equalizer(images):
	'''
	vide equalizer preposessing
	'''
	histo = np.histogram(images, 256, (0, 255))[0]
	cdf = histo.cumsum()
	cdf_eq = (256-1)*(cdf/float(2304))
	cq = cdf_eq[images]
	return cq
