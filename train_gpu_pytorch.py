import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Загрузка и предобработка данных MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)


# Модель
class SimplePyTorchModel(nn.Module):
    def __init__(self):
        super(SimplePyTorchModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.linear(x)


model_gpu = SimplePyTorchModel()

# Использование PyTorch CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_gpu.parameters(), lr=0.01)


# Обучение
def train_pytorch_model(model, trainloader, device):
    model.to(device)
    model.train()  # Установка модели в режим обучения
    start_time = time.time()
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()

            optimizer.zero_grad()
            outputs = model(inputs)

            # Вычисление потерь
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    end_time = time.time()

    training_time = end_time - start_time
    return training_time


training_time_gpu = train_pytorch_model(model_gpu, trainloader, 'cuda')


# Инференс
def inference_model(model, trainloader, device):
    model.to(device)
    model.eval()  # Установка модели в режим инференса
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time


inference_time_gpu = inference_model(model_gpu, trainloader, 'cuda')

print(f"PyTorch GPU Training time: {training_time_gpu} seconds")
print(f"PyTorch GPU Inference time: {inference_time_gpu} seconds")

with open('results_gpu_pytorch.txt', 'w') as f:
    f.write(f"{training_time_gpu},{inference_time_gpu}")
