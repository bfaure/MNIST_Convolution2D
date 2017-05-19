# PyQt4 modules
import PyQt4
from PyQt4 import QtGui
from PyQt4.QtCore import QThread, QRect
from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import sys
import os
import numpy as np
from math import ceil,floor
from scipy import ndimage

#from keras.models import load_model
#from keras import backend as K

#from main import image_t
import main
from main import *

SEARCH_DISTANCE = 6

def get_hit_proximity(x, y, x_vals, y_vals):

	x_search_range = []
	y_search_range = []

	for i in range(SEARCH_DISTANCE*2):
		x_search_range.append(SEARCH_DISTANCE+x-i)
		y_search_range.append(SEARCH_DISTANCE+y-i)

	x_index = -SEARCH_DISTANCE
	best = 1000
	for x_targ in x_search_range:
		y_index = -SEARCH_DISTANCE
		for y_targ in y_search_range:
			for x_val,y_val in list(zip(x_vals, y_vals)):
				if x_val==x_targ and y_val==y_targ:
					proximity = abs(x_index*y_index)
					if proximity < best:
						best = proximity
			y_index += 1
		x_index += 1

	return best


class execution_thread(QThread):
	# Thread to handle the calculations and inferface with the Keras classification modules.
	def __init__(self):
		QThread.__init__(self)

	def run(self):
		self.cur_model = load_model("model/conv2D_classifier.h5")

	def process_data(self):
		self.emit(SIGNAL("send_update(QString)"), "Constructing Image...")
		
		self.x_pos = self.cur_data.x_pos
		self.y_pos = self.cur_data.y_pos

		#proximity_levels = [150.0, 130.0, 90.0, 45.0, 30.0, 25, 20, 15, 10, 5, 1]
		#proximity_depths = [25.0, 30.0, 40.0, 50.0, 80.0, 100.0, 150.0, 175.0, 200.0, 225.0, 255.0]

		proximity_depths = [80.0, 100.0, 120.0, 160.0, 180.0, 200.0, 220.0, 255.0]
		proximity_levels = [8, 7, 6, 5, 4, 3, 2, 1]

		smallest_x 	= 1000
		largest_x 	= 0
		smallest_y 	= 1000
		largest_y 	= 0

		for x,y in list(zip(self.x_pos, self.y_pos)):
			if x > largest_x:
				largest_x = x
			if x < smallest_x:
				smallest_x = x
			if y > largest_y:
				largest_y = y
			if y < smallest_y:
				smallest_y = y

		translated_x = []
		translated_y = []

		for x,y in list(zip(self.x_pos, self.y_pos)):
			translated_x.append(x-smallest_x)
			translated_y.append(y-smallest_y)

		x_size = largest_x - smallest_x
		y_size = largest_y - smallest_y

		x_scale = int(ceil(float(x_size/20)))
		y_scale = int(ceil(float(y_size/20)))

		#new_image = data.image()
		pixels=[]

		pixels2D = [[0.0]*28]*28

		for x in range(28):
			x_upscaled = (x-4)*x_scale
			
			if x < 5 or x > 25:
				x_skip = True
			else:
				x_skip = False
			
			for y in range(28):
				
				if x_skip:
					pixels.append(0.0)
					#new_image.add_pixel_XY(0.0, x, y)
					#pixels2D[x][y]=0.0
					continue

				if y < 5 or y > 25:
					pixels.append(0.0)
					#new_image.add_pixel_XY(0.0, x, y)
					#pixels2D[x][y]=0.0
					continue
				
				y_upscaled = (y-4)*y_scale

				level = get_hit_proximity(x_upscaled, y_upscaled, translated_x, translated_y)
				value = 0.0

				for levels, depths in list(zip(proximity_levels, proximity_depths)):

					if level <= levels:
						value = depths

				#new_image.add_pixel_XY(value, x, y)
				pixels.append(value)
				#pixels2D[x][y]=value

		#for y in range(28):
		#	for x in range(28):
		#		pixels.append(pixels2D[y][x])

		new_image = image_t(pixels)
		
		#for i in range(len(new_image.pixels)):
		#	new_image[i] = list(reversed(new_image.pixels[i]))

		new_image.output_terminal()
		new_image.label = 0

		if K.image_data_format()=='channels_first':
			input_shape = (1,28,28)
		else:
			input_shape = (28,28,1)

		new_image.normalize()

		image = np.array([np.array(new_image.pixels).reshape(input_shape)])
		#proba = self.cur_model.predict_proba(image)
		prediction = self.cur_model.predict(image,batch_size=1,verbose=1)
		print(prediction)
		self.emit(SIGNAL("send_update(QString)"),"Digit is NULL")
		return

		highest_prob = 0.0
		cur_index = 0
		highest_prob_index = 0
		for probability in proba[0]:
			if probability > highest_prob:
				highest_prob = probability
				highest_prob_index = cur_index
			cur_index += 1

		self.emit(SIGNAL("send_update(QString)"), "Digit is a "+str(highest_prob_index)+" with probability of "+str(highest_prob))
		return
		
		
	def get_data(self, path):
		# Need to figure out the bounds of the image (the maximums in all directions)
		self.cur_data = path
		self.process_data()

class drawing_path():
	def __init__(self):
		self.x_pos = []
		self.y_pos = []
	def add_point(self, x, y):
		# Adds a single point to the path
		self.x_pos.append(x)
		self.y_pos.append(y)
	def clear_path(self):
		# Clears both the x and y components of the path
		self.x_pos = []
		self.y_pos = []
	def print_path(self):
		# Outputs a represenation of the path to the terminal
		smallest_x 	= 1000
		largest_x 	= 0
		smallest_y 	= 1000
		largest_y 	= 0

		for x,y in list(zip(self.x_pos, self.y_pos)):
			if x > largest_x:
				largest_x = x
			if x < smallest_x:
				smallest_x = x
			if y > largest_y:
				largest_y = y
			if y < smallest_y:
				smallest_y = y

		translated_x = []
		translated_y = []

		for x,y in list(zip(self.x_pos, self.y_pos)):
			translated_x.append(x-smallest_x)
			translated_y.append(y-smallest_y)

		x_size = largest_x - smallest_x
		y_size = largest_y - smallest_y

		for y in range(y_size):
			line = ""
			for x in range(x_size):
				isPixel = False
				for x_coor,y_coor in list(zip(translated_x, translated_y)):
					if isPixel == False:
						if x_coor==x and y_coor==y:
							line += "X"
							isPixel = True
				if isPixel == False:
					line += "  "
			print(line)


class window(QtGui.QWidget):
	# Window to allow user to input hand written digits for the system to analyze.
	# Basic idea is I am going to create a widget to allow the user to write in a digit
	# and when the user is done the system will gather the user input and send a signal to
	# a slot in the execution_thread which will allow it to run the Keras model and send back
	# a prediction of the digit classification.
	def __init__(self, parent=None):
		super(window, self).__init__()

		self.initThread()

	def initThread(self):
		# Initializes the thread and starts its execution (loading in the model)
		self.thread = execution_thread()
		self.thread.start()
		self.initUI()

	def initUI(self):
		# Initializes the GUI
		self.setFixedHeight(600)
		self.setFixedWidth(450)
		self.setWindowTitle("Dynamic Digit Prediction")
		self.hasDrawing = False
		self.mouseHeld = False

		self.path = drawing_path()

		self.main_layout = QtGui.QVBoxLayout(self) # Main layout for the GUI

		self.rect = QRect(0, 50, 400, 400)

		#self.drawing = QtGui.QPainter(self) # Device to allow user input
		#self.drawing.mousePressEvent.connect(self.start_drawing) # User presses mouse button
		#self.drawing.mouseMoveEvent.connect(self.drawing_occured) # User moving the mouse
		#self.drawing.mouseReleaseEvent.connect(self.end_drawing) # User lets go of mouse button

		self.label = QtGui.QLabel("Click and hold the left mouse button to draw a digit (0-9)", self)
		self.label.move(5, 10)
		self.label.setFixedWidth(300)
		self.results = QtGui.QLabel("Results will appear here", self)
		self.results.move(25, 540)
		self.results.setFixedWidth(300)
		self.result_label = QtGui.QLabel("", self)
		self.result_label.move(330, 490)


		self.clear_button = QtGui.QPushButton("Clear", self)
		self.clear_button.move(330, 535)
		self.clear_button.clicked.connect(self.clear)

		self.upper_line = QtGui.QFrame(self)
		self.upper_line.setFrameShape(QFrame.HLine)
		self.upper_line.move(25, 85)
		self.upper_line.setFixedWidth(400)

		self.lower_line = QtGui.QFrame(self)
		self.lower_line.setFrameShape(QFrame.HLine)
		self.lower_line.move(25, 485)
		self.lower_line.setFixedWidth(400)

		self.left_line = QtGui.QFrame(self)
		self.left_line.setFrameShape(QFrame.VLine)
		self.left_line.move(-25, 100)
		self.left_line.setFixedHeight(400)

		self.right_line = QtGui.QFrame(self)
		self.right_line.setFrameShape(QFrame.VLine)
		self.right_line.move(375, 100)
		self.right_line.setFixedHeight(400)


		QtCore.QObject.connect(self, QtCore.SIGNAL("send_data(PyQt_PyObject)"), self.thread.get_data)
		QtCore.QObject.connect(self.thread, QtCore.SIGNAL("send_update(QString)"), self.update_label)

		self.show()

	def clear(self):
		self.path.clear_path()
		self.update()

	def mousePressEvent(self, event):
		x = event.x()
		y = event.y()
		self.path.clear_path()

		if 100 < y < 500:
			if 25 < x < 425:
				if self.hasDrawing == True:
					self.path.clear()
				self.mouseHeld = True

				position = event.pos()
				
				self.path.add_point(x,y)

				self.results.setText("Position = "+str(position))
				return
			else:
				self.results.setText("Position out of range")
				self.mouseHeld = False
				return
		self.mouseHeld = False
		self.results.setText("Position out of range")
		return

	def mouseMoveEvent(self, event):
		x = event.x()
		y = event.y()
		if 100 < y < 500:
			if 25 < x < 425:
				if self.mouseHeld == True:

					position = event.pos()
					self.path.add_point(x,y)
					self.results.setText("Position = "+str(position))
					self.update()
					return
				else:
					return
			else:
				self.results.setText("Position out of range")
		else:
			self.results.setText("Position out of range")

	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)

		last_x = 0
		last_y = 0
		for x,y in list(zip(self.path.x_pos, self.path.y_pos)):
			if last_x == 0:
				last_x = x
				last_y = y
			else:
				painter.drawLine(last_x, last_y, x, y)
				last_x = x
				last_y = y
		#painter.drawLine(self.last_x, self.last_y, self.cur_x, self.cur_y)
		painter.end()

	def mouseReleaseEvent(self, event):
		self.mouseHeld = False
		self.results.setText("Processing Data...")
		self.emit(SIGNAL("send_data(PyQt_PyObject)"), self.path)
		#self.path.clear_path()

	def update_label(self, text):
		self.results.setText(text)


def main():

	app = QtGui.QApplication(sys.argv)
	_ = window()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()