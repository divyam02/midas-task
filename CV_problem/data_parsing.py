import matplotlib.pyplot as plt
import pickle
import numpy as np

def get_raw_data():
	"""
	Return raw labels and training data(numpy)
	"""
	with open('train_label.pkl', 'rb') as f:
		train_label = pickle.load(f)

	with open('train_image.pkl', 'rb') as f:
		train_data = pickle.load(f)

	print(np.unique(np.asarray(train_label)))

	return (train_label, np.asarray(train_data))

def get_visuals(train_data):
	"""
	@Input:
		numpy array with all the training data

	@Output:
		random data images
	"""
	"""
	for i in train_data:
		i = i.reshape((28, 28))
		print(i.shape)
	"""

	for i in range(5):
		j = train_data[np.random.randint(8000)]
		plt.imshow(j.reshape((28, 28)))
		plt.show()

def get_normalized_data(train_data):

	mean = np.mean(train_data, axis=0)
	print(np.mean(train_data))
	plt.imshow(mean.reshape((28, 28)))
	plt.show()

	std = np.std(train_data, axis=0)
	print(np.std(train_data))
	norm_train_data = train_data - mean
	norm_train_data /= std

	plt.imshow(train_data[0].reshape((28, 28)))

	plt.show()

	plt.imshow(((norm_train_data + np.abs(np.min(norm_train_data, axis=0)))[0].reshape((28, 28))))

	plt.show()

	return norm_train_data	

if __name__ == '__main__':
	l, t = get_raw_data()

	#get_visuals(t)

	#get_normalized_data(t)