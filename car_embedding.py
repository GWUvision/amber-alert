import keras
import tensorflow as tf
import numpy as np
import os
import cv2

from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.models import Model

from sklearn.model_selection import train_test_split

import helpers

IMAGES_DIR = 'resized_images/'
DATA_LIMIT = 0.1


def get_data():
	anchor_images = []
	positive_images = []
	negative_images = []

	files_list = os.listdir(IMAGES_DIR)
	files_list.sort()
	n_images = len(files_list)
	if DATA_LIMIT is not None:
		n_images = int(DATA_LIMIT * n_images)

	i = 0

	for i in range(0, n_images/3):
		# Read 2 images next to each other as anchor + positive
		anchor = cv2.imread(IMAGES_DIR + files_list[2*i], cv2.IMREAD_COLOR)
		positive = cv2.imread(IMAGES_DIR + files_list[2*i + 1], cv2.IMREAD_COLOR)
		
		# Go 2/3 down the list, and increment one by 1
		negative = cv2.imread(IMAGES_DIR + files_list[i + 2*n_images/3], cv2.IMREAD_COLOR)
		
		anchor_images.append(anchor)
		positive_images.append(positive)
		negative_images.append(negative)
		

		print('A: {}\tP: {}\tN: {}'.format(files_list[2*i], files_list[2*i+1], files_list[i+2*n_images/3]))

	
	anchor_np = np.array(anchor_images)
	positive_np = np.array(positive_images)
	negative_np = np.array(negative_images)
	
	stack_data = np.hstack((anchor_np, positive_np, negative_np))

	labels = range(0, n_images/3)
	
	return train_test_split(stack_data, labels, test_size=0.1)


def triplet_loss(y_true, y_pred):
	size = y_pred.shape[1] / 3

	anchor = y_pred[:,0:size]
	positive = y_pred[:,size: 2 * size]
	negative = y_pred[:,2 * size: 3 * size]
	alpha = 0.2

	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
	return loss

def create_model():
	base_model = ResNet50(weights='imagenet', include_top=False)

	anchor_input = Input(shape=(helpers.get_image_shape()), dtype='float32', name = 'anchor')
	positive_input = Input(shape=(helpers.get_image_shape()), dtype='float32', name = 'positive')
	negative_input = Input(shape=(helpers.get_image_shape()), dtype='float32', name = 'negative')

	anchor_out = base_model(anchor_input)	
	positive_out = base_model(positive_input)	
	negative_out = base_model(negative_input)

	merged_vector = concatenate([anchor_out, positive_out, negative_out], axis=-1)	

	model = Model(inputs = [anchor_input, positive_input, negative_input],
              outputs = merged_vector)

	model.compile(optimizer=Adam(), loss = triplet_loss)
	return model

def train():
	X_train, X_test, y_train, y_test = get_data()

	BATCH_SIZE = 8

	X_train = np.hsplit(X_train, 3)
	X_test = np.hsplit(X_test, 3)
	print('shape x before:', np.array(X_train).shape)
	
	# Calculate rounded length divisble by BATCH_SIZE
	X_train_len = len(X_train[0])
	end_i_train = BATCH_SIZE*int(X_train_len/BATCH_SIZE)
	print('end index train', end_i_train)
	
	# Round to length divisible by BATCH_SIZE
	for i in range(0, len(X_train)):
		X_train[i] = np.delete(X_train[i], slice(end_i_train, X_train_len), 0)
	print('shape x after:', np.array(X_train).shape)
	
	y_train = y_train[0:end_i_train]
	print('shape y after:', np.array(y_train).shape)
	
	# Do the same rounding for test data
	X_test_len = len(X_test[0])
	end_i_test = BATCH_SIZE*int(X_test_len/BATCH_SIZE)
	print('end index test', end_i_test)

	for i in range(0, len(X_test)):
		X_test[i] = np.delete(X_test[i], slice(end_i_test, X_test_len), 0)
	print('shape x after:', np.array(X_test).shape)
	
	y_test = y_test[0:end_i_test]
	print('shape y after:', np.array(y_test).shape)
		
	
	model = create_model()
	model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1)

	test_loss = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
	print('Test loss: ', test_loss)

tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 4, "Batch size during evaluation")
train()

