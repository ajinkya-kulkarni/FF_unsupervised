
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from tqdm.auto import tqdm
import shutil
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch import tensor, Tensor

######################################################################################

os.system('cls || clear')

######################################################################################

epochs = 10
batch_size = 64
n_neurons = 2000
n_classes = 10
n_layers = 4
input_size = 28 * 28
n_hid_to_log = 3

######################################################################################

def clean_repo():

	folder_path = "MNIST"
	if os.path.exists(folder_path):
		shutil.rmtree(folder_path)

	file_path = "transformed_dataset.pt"
	if os.path.exists(file_path):
		os.remove(file_path)

######################################################################################

def prepare_data():

	# Define the transform function
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

	# Load the train MNIST dataset
	train_mnist_dataset = torchvision.datasets.MNIST(root="./", train=True, transform=transform, download=True)
	n_train_samples = len(train_mnist_dataset)
	# Load the test MNIST dataset
	test_mnist_dataset = torchvision.datasets.MNIST(root="./", train=False, transform=transform, download=True)

	if not os.path.exists("transformed_dataset.pt"):
		random_pairs = np.random.randint(n_train_samples, size=[n_train_samples, 2])
		random_pairs = [(row[0], row[1]) for row in random_pairs]

		# Transform the data
		transformed_dataset = [
			create_negative_image(train_mnist_dataset[pair[0]][0].squeeze(), train_mnist_dataset[pair[1]][0].squeeze())
			for pair in tqdm(random_pairs, desc = 'Preparing Dataset')]

		# Save the transformed images to a folder
		torch.save(transformed_dataset, 'transformed_dataset.pt')

######################################################################################

def create_mask(shape, iterations: int = 10):
	"""
	Create a binary mask as described in (Hinton, 2022): start with a random binary image and then repeatedly blur
	the image with a filter, horizontally and vertically.

	Parameters
	----------
	shape : tuple
		The shape of the output mask (height, width).
	iterations : int
		The number of times to blur the image.

	Returns
	-------
	numpy.ndarray
		A binary mask with the specified shape, containing fairly large regions of ones and zeros.
	"""

	blur_filter_1 = np.array(((0, 0, 0), (1/4, 1/2, 1/4), (0, 0, 0)))
	blur_filter_2 = blur_filter_1.T

	# Create a random binary image
	image = np.random.randint(0, 2, size=shape)

	# Blur the image with the specified filter
	for i in range(iterations):
		image = np.abs(convolve2d(image, blur_filter_1, mode='same') / blur_filter_1.sum())
		image = np.abs(convolve2d(image, blur_filter_2, mode='same') / blur_filter_2.sum())

	# Binarize the blurred image, i.e. threshold it at 0.5
	mask = np.round(image).astype(np.uint8)

	return tensor(mask)

######################################################################################

def create_negative_image(image_1: Tensor, image_2: Tensor):
	"""
	Create a negative image by combining two images with a binary mask.

	Parameters:
	image_1 (Tensor): The first image to be combined.
	image_2 (Tensor): The second image to be combined.

	Returns:
	Tensor: The negative image created by combining the two input images.

	Raises:
	AssertionError: If the shapes of `image_1` and `image_2` are not the same.

	Examples:
	>>> image_1 = np.random.randint(0, 2, size=(5, 5))
	>>> image_2 = np.random.randint(0, 2, size=(5, 5))
	>>> create_negative_image(image_1, image_2)
	array([[0 0 0 0 1]
		[1 1 0 1 1]
		[0 0 0 1 1]
		[0 1 1 1 0]
		[1 1 0 0 1]])
	"""

	assert image_1.shape == image_2.shape, "Incompatible images and mask shapes."

	mask = create_mask((image_1.shape[0], image_1.shape[1]))

	image_1 = torch.mul(image_1, mask)
	image_2 = torch.mul(image_2, 1 - mask)

	return torch.add(image_1, image_2)

######################################################################################

def goodness_score(pos_acts, neg_acts, threshold=2):
	"""
	Compute the goodness score for a given set of positive and negative activations.

	Parameters:

	pos_acts (torch.Tensor): Numpy array of positive activations.
	neg_acts (torch.Tensor): Numpy array of negative activations.
	threshold (int, optional): Threshold value used to compute the score. Default is 2.

	Returns:

	goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
	score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
	quantity but without the threshold subtraction
	"""

	pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
	neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
	return torch.add(pos_goodness, neg_goodness)

######################################################################################

def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    return dict(accuracy_score=acc)

def ff_layer_init(in_features, out_features, n_epochs, bias, device):
    layer = nn.Linear(in_features, out_features, bias=bias)
    layer.n_epochs = n_epochs
    layer.opt = torch.optim.Adam(layer.parameters())
    layer.goodness = goodness_score
    layer.to(device)
    layer.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)
    return layer

def ff_train(layer, pos_acts, neg_acts, goodness):
    layer.opt.zero_grad()
    goodness = goodness(pos_acts, neg_acts)
    goodness.backward()
    layer.opt.step()

def ff_forward(layer, input):
    input = layer(input)
    input = layer.ln_layer(input.detach())
    return input

def unsupervised_ff_init(n_layers, bias, n_classes, n_hid_to_log, device, n_neurons, input_size, n_epochs):
    model = nn.Module()
    model.n_hid_to_log = n_hid_to_log
    model.n_epochs = n_epochs
    model.device = device

    ff_layers = [ff_layer_init(in_features=input_size if idx == 0 else n_neurons,
                               out_features=n_neurons,
                               n_epochs=n_epochs,
                               bias=bias,
                               device=device)
                 for idx in range(n_layers)]

    model.ff_layers = ff_layers
    model.last_layer = nn.Linear(in_features=n_neurons * n_hid_to_log, out_features=n_classes, bias=bias)

    model.to(device)
    model.opt = torch.optim.Adam(model.last_layer.parameters())
    model.loss = torch.nn.CrossEntropyLoss(reduction="mean")
    return model

def train(model: nn.Module, pos_dataloader: DataLoader, neg_dataloader: DataLoader, goodness_score: float) -> list[float]:
    train_ff_layers(model, pos_dataloader, neg_dataloader, goodness_score)
    return train_last_layer(model, pos_dataloader)

def train_ff_layers(model: nn.Module, pos_dataloader: DataLoader, neg_dataloader: DataLoader, goodness_score: float):
    for epoch in tqdm(range(model.n_epochs), desc = 'Training1'):
        for pos_data, neg_imgs in zip(pos_dataloader, neg_dataloader):
            pos_imgs, _ = pos_data
            pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], 1, -1)).to(model.device)
            neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], 1, -1)).to(model.device)

            for idx, layer in enumerate(model.ff_layers):
                pos_acts = ff_forward(layer, pos_acts)
                neg_acts = ff_forward(layer, neg_acts)
                ff_train(layer, pos_acts, neg_acts, goodness_score)

def train_last_layer(model: nn.Module, dataloader: DataLoader) -> list[float]:
    loss_list = []
    for epoch in tqdm(range(model.n_epochs), desc = 'Training2'):
        epoch_loss = 0
        for images, labels in dataloader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            model.opt.zero_grad()
            preds = unsupervised_ff_forward(model, images)
            loss = model.loss(preds, labels)
            epoch_loss += loss
            loss.backward()
            model.opt.step()
        loss_list.append(epoch_loss / len(dataloader))
    return [l.detach().cpu().numpy() for l in loss_list]

def unsupervised_ff_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    image = image.to(model.device)
    image = torch.reshape(image, (image.shape[0], 1, -1))
    concat_output = []
    for idx, layer in enumerate(model.ff_layers):
        image = ff_forward(layer, image)
        if idx > len(model.ff_layers) - model.n_hid_to_log - 1:
            concat_output.append(image)
    concat_output = torch.cat(concat_output, 2)
    logits = model.last_layer(concat_output)
    return logits

def evaluate(model: nn.Module, dataloader: DataLoader) -> tuple[float, float]:
    nn.Module.eval(model)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = unsupervised_ff_forward(model, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total, correct

def unsupervised_ff_forward(model, image):
    image = image.to(model.device)
    image = torch.reshape(image, (image.shape[0], 1, -1))
    concat_output = []
    for idx, layer in enumerate(model.ff_layers):
        image = ff_forward(layer, image)
        if idx > len(model.ff_layers) - model.n_hid_to_log - 1:
            concat_output.append(image)
    concat_output = torch.cat(concat_output, 2)
    logits = model.last_layer(concat_output)
    return logits.squeeze()


######################################################################################

def plot_loss(loss):
	# plot the loss over epochs
	fig = plt.figure()
	plt.plot(list(np.int_(range(len(loss)))), loss)
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("Loss Plot")
	plt.savefig("Loss Plot.png", bbox_inches='tight', dpi = 200)
	plt.close()

######################################################################################

if __name__ == '__main__':

	file_path = "Loss Plot.png"
	if os.path.exists(file_path):
		os.remove(file_path)

	clean_repo()

	print()
	
	prepare_data()

	print()

	# Load the MNIST dataset
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

	pos_dataset = torchvision.datasets.MNIST(root='./', download=False, transform=transform, train=True)

	# Create the data loader
	pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	# Load the transformed images
	neg_dataset = torch.load('transformed_dataset.pt')

	# Create the data loader
	neg_dataloader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	# Load the test images
	test_dataset = torchvision.datasets.MNIST(root='./', train=False, download=False, transform=transform)

	# Create the data loader
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	unsupervised_ff = unsupervised_ff_init(n_layers=n_layers, bias=True, n_classes=n_classes, n_hid_to_log=n_hid_to_log, device=device, n_neurons=n_neurons, input_size=input_size, n_epochs=epochs)

	loss = train(unsupervised_ff, pos_dataloader, neg_dataloader, goodness_score)

	plot_loss(loss)

	accuracy_train, correct_train = evaluate(unsupervised_ff, pos_dataloader)
	print(f"Train accuracy: {accuracy_train * 100:.2f}% ({correct_train} out of {len(pos_dataloader.dataset)})")

	accuracy_test, correct_test = evaluate(unsupervised_ff, test_dataloader)
	print(f"Test accuracy: {accuracy_test * 100:.2f}% ({correct_test} out of {len(test_dataloader.dataset)})")

	clean_repo()

	print()

######################################################################################

