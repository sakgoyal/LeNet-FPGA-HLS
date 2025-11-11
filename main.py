import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qat_qconfig,
    prepare_qat,
)
from torch.nn.quantized import Conv2d as QConv2d
from torch.nn.quantized import Linear as QLinear
from torch.quantization import QConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

# Suppress the specific deprecation warning
warnings.filterwarnings(
    "ignore",
    message="torch.ao.quantization is deprecated.*",
    category=DeprecationWarning,
)

class LeNet(nn.Module):
    qconfig: QConfig | None = None
    def __init__(self):
        super(LeNet, self).__init__()
        self.quant = QuantStub()

        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10, bias=False)

        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)

        x = self.dequant(x)
        return x

def export_model_to_c_files(model: LeNet):
    print("Exporting quantized model parameters to C files")

    layer_mapping = {
        'conv1': 'c1_weights.bin',
        'conv2': 'c3_weights.bin',
        'fc1': 'f5_weights.bin',
        'fc2': 'f6_weights.bin',
        'fc3': 'output_weights.bin'
    }

    for name, module in model.named_modules():
        if isinstance(module, (QLinear, QConv2d)):
            if name not in layer_mapping:
                continue

            print(f"Exporting layer: {name} -> {layer_mapping[name]}")

            weights_int = module.weight().int_repr().cpu().numpy()

            output_path = Path(layer_mapping[name])
            with open(output_path, 'w') as f:
                array_str = str(weights_int.tolist())
                array_str = array_str.replace('[', '').replace(']', '').replace(' ', '')
                f.write(array_str)

    print("C file export complete.")

def main(num_epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LeNet().to(device)

    model.qconfig = get_default_qat_qconfig('fbgemm')

    print("Fusing modules...")
    model.eval()
    torch.quantization.fuse_modules(model, [
        ['conv1', 'relu1'],
        ['conv2', 'relu2'],
        ['fc1', 'relu3'],
        ['fc2', 'relu4']
    ], inplace=True)

    print("Preparing model for QAT...")
    model.train()
    model_qat = prepare_qat(model)
    model_qat.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_qat.parameters(), lr=1e-3)

    print(f"Starting QAT for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_qat(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training complete.")

    print("Converting model to INT8...")
    model_qat.eval()
    model_qat.to('cpu')

    model_int8 = convert(model_qat)

    print("INT8 model conversion successful.")

    print("\n--- Final INT8 Model Architecture ---")
    print(model_int8)

    torch.save(model_int8.state_dict(), "lenet_int8.pth")
    print("\nSaved INT8 model state_dict to 'lenet_int8.pth'")


    export_model_to_c_files(model_int8)


def generate_samples(num_samples: int = 10):
    train_dataset = MNIST(root='./data', download=True)
    for i in range(num_samples):
        image, label = train_dataset[i]
        image_data = list(image.getdata())
        image_data = [pixel - 128 for pixel in image_data]

        Path('data/MNIST/samples').mkdir(parents=True, exist_ok=True)

        with open(f'data/MNIST/samples/image_{i}.bin', 'wb') as f:
            image_str = ','.join(str(pixel) for pixel in image_data)
            f.write(image_str.encode('utf-8'))
        with open(f'data/MNIST/samples/label_{i}.bin', 'w') as f:
            f.write(str(label))

if __name__ == '__main__':
    main(num_epochs=10)
    generate_samples(num_samples=100)
