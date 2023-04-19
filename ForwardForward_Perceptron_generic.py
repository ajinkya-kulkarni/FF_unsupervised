import torch
import time
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import os
os.system('clear')

print()

def one_hot_encode_cifar10(img0, lab):
    num_classes = 10 # assuming there are 10 possible classes
    img = torch.zeros(img0.shape[0], img0.shape[1] + num_classes, img0.shape[2], img0.shape[3]).to(img0.device)
    img[:, :img0.shape[1], :, :] = img0
    for i in range(img0.shape[0]):
        img[i, img0.shape[1] + lab[i], :, :] = 1.0
    return img


# Load CIFAR-10 Data
train_loader = DataLoader(
	CIFAR10('./CIFAR10_data/', train=True,
			download=True,
			transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Lambda(lambda x: x.permute(1, 2, 0))])),
	batch_size=50000)

test_loader = DataLoader(
	CIFAR10('./CIFAR10_data/', train=False,
			download=True,
			transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Lambda(lambda x: x.permute(1, 2, 0))])),
	batch_size=10000)

print()

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training images
img0, lab = next(iter(train_loader))
img0 = img0.to(device)
lab = lab.to(device)

# Validation images
img0_tst, lab_tst = next(iter(test_loader))
img0_tst = img0_tst.to(device)
lab_tst = lab_tst.to(device)

# Forward Forward Applied to a Single Perceptron for CIFAR-10 Classification
n_input, n_out1, n_out2 = 32*32*3 + 32*3, 1000, 100  # Input dimensions adjusted for CIFAR-10 images with the encoded labels

batch_size, learning_rate = 1000, 1e-4
g_threshold = 10
epochs = 1000

perceptron = torch.nn.Sequential(
	torch.nn.Linear(n_input, n_out1, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(n_out1, n_out2, bias=True),
	torch.nn.ReLU()
)

perceptron.to(device)
optimizer = torch.optim.Adam(perceptron.parameters(), lr=learning_rate)

N_trn = img0.size(0)  # Use all training images (50000)

for epoch in range(epochs):
	img = img0.clone()
	perm = torch.randperm(N_trn)
	img = img[perm]
	lab = lab[perm]

	img_encoded = one_hot_encode_cifar10(img, lab)

	L_tot = 0

	for i in tqdm(range(0, N_trn, batch_size), desc = 'Training'):
		perceptron.zero_grad()

		img_batch = img_encoded[i:i + batch_size].view(batch_size, -1)
		lab_batch = lab[i:i + batch_size]

		output = perceptron(img_batch)
		loss = torch.nn.functional.cross_entropy(output, lab_batch)
		L_tot += loss.item()  # Accumulate total loss for epoch

		loss.backward()  # Compute gradients
		optimizer.step()  # Update parameters

	# Test model with validation set
	N_tst = img0_tst.size(0)  # Use all test images (10000)

	img_encoded_tst = one_hot_encode_cifar10(img0_tst, lab_tst)
	output_tst = perceptron(img_encoded_tst.view(N_tst, -1))
	predicted_label = torch.argmax(output_tst, dim=1)

	# Count number of correctly classified images images in validation set
	Ncorrect = (predicted_label == lab_tst).sum().cpu().numpy()

	print("Epoch", epoch+1, " Test Error", round(100 - Ncorrect/N_tst*100, 2), "%")

print()