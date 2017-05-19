
from PyQt4 import QtGui,QtCore 
from PyQt4.QtCore import * 
from PyQt4.QtGui import * 

import sys, os
import numpy as np

import main
from main import *

class predict_worker(object):
	def __init__(self,parent=None,model_f="model/conv2D_classifier.h5"):
		self.cur_model = load_model(model_f)
		self.parent=parent

	def process_data(self,data):
		self.cur_data = data
		self.parent.results.setText("Calculating...")
		
		self.x_pos = self.cur_data.x_pos
		self.y_pos = self.cur_data.y_pos

		buf = 4 # 4 buffer on all sides

		max_x = max(self.x_pos)
		min_x = min(self.x_pos)
		max_y = max(self.y_pos)
		min_y = min(self.y_pos)

		x_span = float(max_x-min_x)
		y_span = float(max_y-min_y) 

		for i in range(len(self.x_pos)):
			self.x_pos[i] = (self.x_pos[i]-min_x)/x_span*20.0+buf
			self.y_pos[i] = (self.y_pos[i]-min_y)/y_span*20.0+buf

		pixels = []

		for y in range(28):
			for x in range(28):
				f_len = len(pixels)
				for p_x,p_y in zip(self.x_pos,self.y_pos):
					if int(p_x) in [x-1,x,x+1] and int(p_y) in [y-1,y,y+1]:
						pixels.append(1.0)
						break
				if f_len==len(pixels):
					pixels.append(0.0)

		image = image_t(pixels)
		image.output_terminal(threshold=0.9)

		if K.image_data_format()=='channels_first':
			input_shape = (1,28,28)
		else:
			input_shape = (28,28,1)

		image = np.array([np.array(image.pixels).reshape(input_shape)])
		prediction = self.cur_model.predict(image,batch_size=1,verbose=0)
		
		highest_prob = 0.0 
		highest_prob_val = None

		i=0
		for p in prediction[0]:
			if p>highest_prob:
				highest_prob=p 
				highest_prob_val=i
			i+=1

		self.parent.results.setText("Digit is "+str(highest_prob_val)+", prob = %0.2f"%(100.0*highest_prob))

class drawing_path():
	def __init__(self):
		self.x_pos = []
		self.y_pos = []

	def add_point(self, x, y):
		self.x_pos.append(x)
		self.y_pos.append(y)

	def clear_path(self):
		self.x_pos = []
		self.y_pos = []

	def print_path(self):
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

class window(QWidget):
	def __init__(self, parent=None):
		super(window, self).__init__()
		self.worker = predict_worker(parent=self)
		self.init_ui()

	def init_ui(self):
		self.setFixedHeight(600)
		self.setFixedWidth(450)
		self.setWindowTitle("MNIST Digit Prediction")
		self.hasDrawing = False
		self.mouseHeld = False

		self.path = drawing_path()

		self.rect = QRect(0, 50, 400, 400)

		self.label = QLabel("Click and hold the left mouse button to draw.", self)
		self.label.move(25, 10)
		self.label.setFixedWidth(300)

		self.label2 = QLabel("Classifications include numerals (0-9).",self)
		self.label2.move(25,35)
		self.label2.setFixedWidth(300)

		self.results = QLabel("Results will appear here", self)
		self.results.move(25, 540)
		self.results.setFixedWidth(300)
		self.result_label = QLabel("", self)
		self.result_label.move(330, 490)

		self.clear_button = QPushButton("Clear", self)
		self.clear_button.move(330, 535)
		self.clear_button.clicked.connect(self.clear)

		self.upper_line = QFrame(self)
		self.upper_line.setFrameShape(QFrame.HLine)
		self.upper_line.move(25, 85)
		self.upper_line.setFixedWidth(400)

		self.lower_line = QFrame(self)
		self.lower_line.setFrameShape(QFrame.HLine)
		self.lower_line.move(25, 485)
		self.lower_line.setFixedWidth(400)

		self.left_line = QFrame(self)
		self.left_line.setFrameShape(QFrame.VLine)
		self.left_line.move(-25, 100)
		self.left_line.setFixedHeight(400)

		self.right_line = QFrame(self)
		self.right_line.setFrameShape(QFrame.VLine)
		self.right_line.move(375, 100)
		self.right_line.setFixedHeight(400)

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
		if len(self.path.x_pos)<4: return
		self.results.setText("Processing Data...")
		self.worker.process_data(self.path)

	def update_label(self, text):
		self.results.setText(text)


def main():

	app = QtGui.QApplication(sys.argv)
	_ = window()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()