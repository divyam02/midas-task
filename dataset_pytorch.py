from torch.utils.data.datasets import Dataset
import torch

class fmnist(Dataset):
	def __init__(self, path, height, width, transforms):
		"""
		Return raw labels and training data(numpy)
		"""
		with open('train_label.pkl', 'rb') as f:
			self.labels = np.asarray(pickle.load(f))

		with open('train_image.pkl', 'rb') as f:
			self.data = np.asarray(pickle.load(f))

		self.transforms = transforms
		self.height = height
		self.width = width

	def __getitem__(self, index):
		"""
		"""
		single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
		# Convert image from numpy array to PIL image, mode 'L' is for grayscale
        
        img_as_img = Image.fromarray(img_as_np)
        
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

	def __len__(self):
		return len(self.label) # of number of examples

