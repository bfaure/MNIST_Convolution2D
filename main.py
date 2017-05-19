
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

import numpy as np 
	
test_images_f  = "data/t10k-images.idx3-ubyte"
test_labels_f  = "data/t10k-labels.idx1-ubyte"

train_images_f = "data/train-images.idx3-ubyte"
train_labels_f = "data/train-labels.idx1-ubyte"

class image_t(object):
	def __init__(self,pixels=None,label=None):
		self.pixels = pixels # 2D list of pixel values
		self.label  = label # alphanumeric value

	# print digit to terminal
	def output_terminal(self, threshold=20):
		if len(self.pixels)==784:
			line=[]
			for p in self.pixels:
				if int(p)>threshold: line.append(" X")
				else: line.append("  ")
				if len(line)==28:
					print(''.join(line))
					line=[]
			if self.label!=None: print("\nImage label = %s"%self.label)
		else:
			print("Image not 784 pixels!")

	# divide each entry by 255
	def normalize(self):
		for i in range(len(self.pixels)):
			self.pixels[i] = float(self.pixels[i])/255.0

	# create onehot vector for label
	def onehot_label(self):
		self.onehot = np.zeros(10)
		self.onehot[int(self.label)] = 1.0

# Read in num_words words starting at byte_offset. This is used for
# reading in the metadata of the file at the header.
def read_words(file, byte_offset, num_words):
	# Read a single word from a file and return the decimal representation
	def read_word(file, index):
		vals = []
		for i in range(4):
			file.seek(i+index)
			vals.append(ord(file.read(1)))
		val = vals[0]*(16**6) + vals[1]*(16**4) + vals[2]*(16**2) + vals[3]
		return val
	words = []
	for i in range(num_words):
		val = read_word(file, byte_offset+(i*4))
		words.append(val)
	return words

# Reads in num_bytes bytes starting at byte_offset
def read_bytes(file, byte_offset, num_bytes):
	vals = []
	for i in range(num_bytes):
		file.seek(byte_offset+i)
		vals.append(ord(file.read(1)))
	return vals

def get_images(file, byte_offset, num_images):
	pictures = []
	for i in range(num_images):
		pixels = read_bytes(file,byte_offset+(i*784),784)
		pictures.append(image_t(pixels))
	return pictures

def load_data(max_images=None):
	# load test image set
	with open(test_images_f,'rb') as f:
		magic,num_images,x_range,y_range = read_words(f,0,4)
		test_images = get_images(f,16,num_images if max_images is None else max_images)
	# load test image labels
	with open(test_labels_f,'rb') as f:
		test_labels = read_bytes(f,8,num_images)
	# correlate labels with images
	for t,l in zip(test_images,test_labels):
		t.label = l 
	# sanity check
	test_images[0].output_terminal()

	# load train image set
	with open(train_images_f,'rb') as f:
		magic,num_images,x_range,y_range = read_words(f,0,4)
		train_images = get_images(f,16,num_images if max_images is None else max_images)
	# load train image labels
	with open(train_labels_f,'rb') as f:
		train_labels = read_bytes(f,8,num_images)
	# correlate labels with images
	for t,l in zip(train_images,train_labels):
		t.label = l 
	# sanity check
	train_images[0].output_terminal()

	return test_images,train_images 

def main():
	print("Loading data...")
	test_images,train_images = load_data()

	print("Normalizing data...")
	for t in test_images:
		t.normalize()
		t.onehot_label()
	for t in train_images:
		t.normalize()
		t.onehot_label()

	print("Vectorizing data...")
	test_X=[]
	test_Y=[]
	train_X=[]
	train_Y=[]

	if K.image_data_format()=='channels_first':
		input_shape = (1,28,28)
	else:
		input_shape = (28,28,1)

	for t in test_images:
		test_X.append(np.array(t.pixels).reshape(input_shape))
		test_Y.append(t.onehot)
	for t in train_images:
		train_X.append(np.array(t.pixels).reshape(input_shape))
		train_Y.append(t.onehot)

	test_X = np.array(test_X)
	test_Y = np.array(test_Y)
	train_X = np.array(train_X)
	train_Y = np.array(train_Y)

	print("Building model...")
	num_classes = 10
	batch_size = 128 
	epochs = 20

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	print("Compiling model...")
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	print("Fitting model...")
	model.fit(train_X, train_Y,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(test_X, test_Y))
	score = model.evaluate(test_X, test_Y, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	print("Saving model...")
	model.save("model/conv2D_classifier.h5")

	print("Done.")

if __name__ == '__main__':
	main()