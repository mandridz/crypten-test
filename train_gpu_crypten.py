import crypten
import crypten.nn as cnn
import crypten.optim as crypten_optim
import torch
import torch.nn as nn
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

# Использование PyTorch CrossEntropyLoss
criterion = nn.CrossEntropyLoss()


# Обучение
def train_crypten_model(model, trainloader, device):
    optimizer = crypten_optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs_enc = crypten.cryptensor(inputs.to(device))
            labels_plain = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs_enc)

            # Отладочные выводы размеров и типов
            # print(f"Epoch: {epoch}, Batch size: {inputs.size(0)}")
            # print(f"inputs_enc size: {inputs_enc.size()}, type: {type(inputs_enc)}")
            # print(f"outputs size: {outputs.size()}, type: {type(outputs)}")
            # print(f"labels_plain size: {labels_plain.size()}, type: {type(labels_plain)}")

            # Убедимся, что метки имеют правильный размер
            assert outputs.size(
                1) == 10, f"Размер выходных данных должен быть [batch_size, 10], но получил {outputs.size()}"
            assert labels_plain.size(0) == outputs.size(
                0), f"Размер меток должен быть [batch_size], но получил {labels_plain.size()}"

            # Декриптование выходов для использования в PyTorch функции потерь
            outputs_plain = outputs.get_plain_text()

            # Вычисление потерь
            loss = criterion(outputs_plain, labels_plain)

            # print(f"Loss: {loss.item()}")

            # Конвертация потерь обратно в зашифрованный тензор
            loss_enc = crypten.cryptensor(loss.item(), src=0)

            loss_enc.backward()
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
