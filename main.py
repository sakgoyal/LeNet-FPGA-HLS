from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

torch.set_default_dtype(torch.float32)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, bias=True, dtype=torch.float32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, bias=True, dtype=torch.float32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=True, dtype=torch.float32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=True, dtype=torch.float32)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10, bias=True, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)

        return x

def export_model_to_c_files(model: LeNet, use_float16: bool = False):
    print(f"Exporting {'float16' if use_float16 else 'float32'} model parameters to C files")

    layer_mapping = {
        'conv1': ('c1_weights.bin', 'c1_bias.bin'),
        'conv2': ('c3_weights.bin', 'c3_bias.bin'),
        'fc1': ('f5_weights.bin', 'f5_bias.bin'),
        'fc2': ('f6_weights.bin', 'f6_bias.bin'),
        'fc3': ('output_weights.bin', 'output_bias.bin')
    }

    for name, module in model.named_modules():
        print(f"Processing module: {name} of type {type(module)}")
        if name not in layer_mapping:
            continue
        print(f"Exporting layer: {name} -> {layer_mapping[name]}")

        dtype_str = 'float16' if use_float16 else 'float32'
        weights = module.weight.data.half() if use_float16 else module.weight.data.float()
        bias = (module.bias.data.half() if use_float16 else module.bias.data.float()) if module.bias is not None else None

        weight_path, bias_path = layer_mapping[name]
        with open(Path(weight_path), 'w') as f:
            array_str = str(weights.cpu().numpy().astype(dtype_str).flatten().tolist())
            array_str = array_str.replace('[', '').replace(']', '').replace(' ', '')
            f.write(array_str)
        if bias is not None:
            with open(Path(bias_path), 'w') as f:
                array_str = str(bias.cpu().numpy().astype(dtype_str).flatten().tolist())
                array_str = array_str.replace('[', '').replace(']', '').replace(' ', '')
                f.write(array_str)

    print("C file export complete.")

def main(num_epochs: int = 3):
    device = torch.device("cuda")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = LeNet().to(device)
    model = model.float()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        samples = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use autocast for forward pass
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            samples += images.size(0)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        train_loss = running_loss / samples

        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                eval_loss += loss.item() * images.size(0)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        eval_loss /= total
        eval_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")

    print("Training complete.")

    model.eval()
    model.to('cpu')

    # Convert model to float16 for export
    model_fp16 = model.half()

    torch.save(model.state_dict(), "lenet.pth")
    torch.save(model_fp16.state_dict(), "lenet_fp16.pth")
    print("\nSaved model state_dict to 'lenet.pth' and 'lenet_fp16.pth'")

    # Export float16 quantized model
    export_model_to_c_files(model_fp16, use_float16=True)


def generate_samples(num_samples: int = 10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization - must match training
    ])
    train_dataset = MNIST(root='./data', download=True, transform=transform)
    for i in range(num_samples):
        image, label = train_dataset[i]
        image_data = list(image.flatten().numpy())

        Path('data/MNIST/samples').mkdir(parents=True, exist_ok=True)

        with open(f'data/MNIST/samples/image_{i}.bin', 'w') as f:
            # convert each pixel to string and join with commas
            # image_str = ','.join(str(pixel) for pixel in image_data)

            # add a newline after every 28 values for readability instead of a single line
            image_str = ''
            for row in range(28):
                row_data = image_data[row * 28:(row + 1) * 28]
                row_str = ','.join(str(pixel) for pixel in row_data)
                image_str += row_str + '\n'
            f.write(image_str)
        with open(f'data/MNIST/samples/label_{i}.bin', 'w') as f:
            f.write(str(label))

if __name__ == '__main__':
    main(num_epochs=10)
    generate_samples(num_samples=100)
