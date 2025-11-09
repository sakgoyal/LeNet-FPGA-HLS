import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Suppress the specific deprecation warning
warnings.filterwarnings(
    "ignore",
    message="torch.ao.quantization is deprecated.*",
    category=DeprecationWarning
)

from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qat_qconfig,
    prepare_qat,
)


class LeNet(nn.Module):
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

def export_model_to_json(model, filename):
    print(f"Exporting quantized model parameters to {filename}...")
    export_data = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.quantized.Linear, nn.quantized.Conv2d)):
            print(f"Processing layer: {name}")

            if module.weight().qscheme() == torch.per_tensor_affine:
                weight_scale = module.weight().q_scale()
                weight_zero_point = module.weight().q_zero_point()
            elif module.weight().qscheme() == torch.per_channel_affine:
                weight_scale = module.weight().q_per_channel_scales().cpu().numpy().tolist()
                weight_zero_point = module.weight().q_per_channel_zero_points().cpu().numpy().tolist()
            else:
                print(f"Skipping layer {name} with unknown qscheme: {module.weight().qscheme()}")
                continue

            weights_int = module.weight().int_repr().cpu().numpy().tolist()
            print(module.bias, flush=True)

            output_scale = module.scale
            output_zero_point = module.zero_point

            export_data[name] = {
                'weights_int8': weights_int,
                'weight_scale': weight_scale,
                'weight_zero_point': weight_zero_point,
                'output_scale': output_scale,
                'output_zero_point': output_zero_point
            }

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=4)

    print("Export complete.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LeNet().to(device)

    model.qconfig = get_default_qat_qconfig('qnnpack')

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

    num_epochs = 1
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

    export_model_to_json(model_int8, "lenet_int8_params.json")

if __name__ == '__main__':
    main()
