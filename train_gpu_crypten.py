import crypten
import crypten.nn as cnn
import crypten.optim as crypten_optim
import torch
import torchvision
import torchvision.transforms as transforms
import time

crypten.init()

# Загрузка и предобработка данных MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)


# Модель
class SimpleCrypTenModel(cnn.Module):
    def __init__(self):
        super(SimpleCrypTenModel, self).__init__()
        self.linear = cnn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.linear(x)


model_cryp_gpu = SimpleCrypTenModel()

# Зашифрование модели и перемещение на GPU
model_cryp_gpu.encrypt()
model_cryp_gpu = model_cryp_gpu.to('cuda')


# Обучение
def train_crypten_model(model, trainloader, device):
    criterion = crypten.nn.CrossEntropyLoss()
    optimizer = crypten_optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs_enc = crypten.cryptensor(inputs.to(device))
            labels_plain = labels.to(device).long()

            # Убедимся, что метки имеют правильный размер
            assert inputs_enc.size(0) == labels_plain.size(
                0), f"Размер меток должен быть [batch_size], но получил {labels_plain.size()}"

            optimizer.zero_grad()
            outputs = model(inputs_enc)

            # Проверка формы выходных данных
            assert outputs.size(
                1) == 10, f"Размер выходных данных должен быть [batch_size, 10], но получил {outputs.size()}"

            loss = criterion(outputs, labels_plain)  # Используем незашифрованные метки
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    end_time = time.time()

    training_time = end_time - start_time
    return training_time


training_time_cryp_gpu = train_crypten_model(model_cryp_gpu, trainloader, 'cuda')


# Инференс
def inference_model(model, trainloader, device):
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs_enc = crypten.cryptensor(inputs.to(device))
            outputs = model(inputs_enc)
    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time


inference_time_cryp_gpu = inference_model(model_cryp_gpu, trainloader, 'cuda')

print(f"CrypTen GPU Training time: {training_time_cryp_gpu} seconds")
print(f"CrypTen GPU Inference time: {inference_time_cryp_gpu} seconds")

with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{training_time_cryp_gpu},{inference_time_cryp_gpu}")
