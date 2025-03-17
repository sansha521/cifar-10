import os
import configparser

import pickle

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import csv
#%%
from torchsummary import summary
#%%
from matplotlib import pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from heapq import heapify, heappush, heappop

import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from heapq import heapify, heappush, heappop

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_batch_size = 128
train_batch_size = 128



# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the config file
config.read('config.ini')

# Access values from the config file
LR = config['train']['learning_rate']
WEIGHT_DECAY = config['train']['weight_decay']
DATAFILE_PATH = config['data']['filepath']
MODEL_NAME = config['model']['name']
MODEL_PATH = config['model']['path']
PREDICTION_FILE_PATH = config['output']['filepath']
MODEL_NAME_WITH_ACCURACY = config['model']['accuracy_name']


# %%
# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Modified BasicBlock with SE
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)  # Adding SE Block

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Modified ResNet with SE and adjusted depth
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 254, num_blocks[2], stride=2)
        self.linear = nn.Linear(254 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Instantiate the model with adjusted num_blocks [4,5,3]
def ResNetSE():
    return ResNet(BasicBlock, [4, 5, 3])


# Training Setup




def transform():
    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


def prepare_data():
    transform_train, transform_test = transform
    # Download and load the training dataset
    trainset = torchvision.datasets.CIFAR10(
        root=DATAFILE_PATH,  # Path to store the dataset
        train=True,  # Load the training set
        download=True,  # Download the dataset if it doesn't exist
        transform=transform_train  # Apply the defined transformation
    )

    # Create a DataLoader for the training set
    trainloader = torch.utils.data.DataLoader(
        trainset,  # Dataset to load
        batch_size=train_batch_size,  # Batch size
        shuffle=True,  # Shuffle the data
        num_workers=4  # Number of subprocesses to use for data loading
    )

    # Download and load the test dataset
    testset = torchvision.datasets.CIFAR10(
        root=DATAFILE_PATH,  # Path to store the dataset
        train=False,  # Load the test set
        download=True,  # Download the dataset if it doesn't exist
        transform=transform_test  # Apply the defined transformation
    )

    # Create a DataLoader for the test set
    testloader = torch.utils.data.DataLoader(
        testset,  # Dataset to load
        batch_size=train_batch_size,  # Batch size
        shuffle=False,  # Do not shuffle the data
        num_workers=4  # Number of subprocesses to use for data loading
    )

    return trainloader, testloader


def set_optimizer(model):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if 'ReduceLR' in MODEL_NAME:
        # Training Config
        # optimizer = SGD(model.parameters(), lr=0.0001, weight_decay=5e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # because you're tracking validation loss
            factor=0.1,  # reduce LR by 10x
            patience=5,  # wait 5 epochs without improvement
            threshold=1e-4,  # minimum change in val_loss to qualify as improvement
            cooldown=2,  # stop LR reduction for 2 epochs after each reduction
            min_lr=1e-5,  # don’t reduce below this
            verbose=True  # print when LR is reduced
        )

    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


    if 'label_smoothing' in MODEL_NAME:
        cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        # Training Config
        cross_entropy_loss = nn.CrossEntropyLoss()


    return optimizer, scheduler, cross_entropy_loss

# %%
# Initialize the model's weights with Xavier initialization
def init_weights_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):  # Check if the layer is Linear or Conv2d
        init.xavier_uniform_(m.weight)  # Apply Xavier uniform initialization to weights
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to zero (optional)


def init_model_weights(model):
    # Apply the initialization function to the model
    model.apply(init_weights_xavier)
    print(f"Model weights initialized with Xavier initialization.")



def train_model():
    model = ResNetSE().to(device)
    optimizer, scheduler, cross_entropy_loss = set_optimizer(model)
    trainloader, testloader = prepare_data()
    init_model_weights(model)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    # %%
    top_5_accuracy = [0]
    # %%
    training_loss_history = []
    validation_loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Training started >>> ")


    # Training Loop
    for epoch in tqdm(range(150)):
        model.train()
        total_loss = 0
        total_validation_loss = 0
        total_correct_train = 0
        total_train = 0
        for input, labels in trainloader:
            input, labels = input.to(device).float().view(-1, 3, 32, 32), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            total_correct_train += (predicted == labels).sum().item()

        scheduler.step()
        epoch_train_loss = total_loss / len(trainloader)
        epoch_train_accuracy = 100 * total_correct_train / total_train

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(trainloader):.3f}", end='\r')
        training_loss_history.append(total_loss / len(trainloader))
        training_accuracy_history.append(epoch_train_accuracy)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device).view(-1, 3, 32, 32), labels.to(device)
                outputs = model(inputs)
                test_loss = cross_entropy_loss(outputs, labels)
                total_validation_loss += test_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        if accuracy > top_5_accuracy[0]:
            if len(top_5_accuracy) >= 5:
                heappop(top_5_accuracy)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'./{MODEL_PATH}/resnet_full_checkpoint_{accuracy}.pth')

            heappush(top_5_accuracy, accuracy)
        # torch.save(model.state_dict(), f'/scratch/ar6316/DeepLearning_Project_1/models/{model_name}_model_weights_accuracy_{(accuracy*100)//1}.pth')

        validation_loss_history.append(total_validation_loss / len(testloader))
        print(f"Test Accuracy: {accuracy:.2f}%")
        validation_accuracy_history.append(accuracy)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




class Cifar10(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()
        label = data.get(b'labels', None)
        self.ids = [int(x) for x in data[b'ids']]
        data = [transform_unseen(x) for x in data[b'data']]
        # self.ids_to_data = {}
        self.x_data = torch.stack(data).to(device)

        # self._construct_id_hash()

        if label is not None:
            self.y_data = torch.tensor(label, dtype=torch.uint8).to(device)
        else:
            self.y_data = None

    def __len__(self):
        return len(self.x_data)  # required

    def __getitem__(self, idx):
        data = self.x_data[idx]
        return data


def visualize(input, counter):
    # Convert the tensor to (H, W, C) format using .permute()
    if counter < 5:
        print(input.shape)
        image_np = input[0].permute(1, 2, 0).cpu().numpy()

        # Plot the image
        plt.imshow(image_np)
        plt.axis('off')  # Optional: turn off axes for better visualization
        plt.show()



def test():
    nolabel_file = './cifar_test_nolabel.pkl' # Hard coded.

    no_label = unpickle(nolabel_file)

    transform_unseen = transforms.Compose([
        transforms.ToTensor(),  # Scale to [0, 1]
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalize
    ])

    load_model_name = MODEL_NAME_WITH_ACCURACY
    model = ResNetSE().to(device)  # Replace with your model class and arguments
    model.load_state_dict(torch.load(f"{MODEL_PATH}/{load_model_name}"))
    model.eval()

    # Prepare unlabeled data and DataLoader
    unlabeled_data = no_label  # Replace with your input data (e.g., images or tensors)
    dataset = Cifar10(unlabeled_data)
    unlabelled_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Run inference on unlabeled data
    counter = 0
    predictions = []
    with torch.no_grad():
        for batch in unlabelled_dataloader:
            # for item in batch:
            #     ids.append(dataset.get_id(item))

            batch = batch.to(device).float().view(-1, 3, 32, 32)
            counter += 1
            visualize(batch, counter)
            outputs = model(batch)
            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())

    final_output = {'ID': [int(x) for x in no_label[b'ids']], 'Labels': [int(x) for x in predictions]}

    with open(f'{PREDICTION_FILE_PATH}/output_{MODEL_NAME}_transformed.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header (keys of the dictionary)
        writer.writerow(final_output.keys())

        # Write the rows (values of the dictionary, zipped together)
        writer.writerows(zip(*final_output.values()))

        # Optional: Print confirmation
        print("CSV file has been created successfully!")

if __name__ == '__main__':
    train_model()
    test()