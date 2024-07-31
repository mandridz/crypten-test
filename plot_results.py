import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
results = {}
files = ['results_cpu_crypten.txt', 'results_gpu_crypten.txt', 'results_cpu_pytorch.txt', 'results_gpu_pytorch.txt']
labels = ['CrypTen CPU', 'CrypTen GPU', 'PyTorch CPU', 'PyTorch GPU']

for file, label in zip(files, labels):
    with open(file, 'r') as f:
        data = f.read().strip().split(',')
        results[label] = [float(x) for x in data]

# Данные для графиков
categories = ['Training Time', 'Inference Time']
n_categories = len(categories)
index = np.arange(n_categories)

# Построение графиков
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8

for i, label in enumerate(labels):
    plt.bar(index + i * bar_width, results[label], bar_width, alpha=opacity, label=label)

plt.xlabel('Metrics')
plt.ylabel('Time (seconds)')
plt.title('Training and Inference Time Comparison')
plt.xticks(index + bar_width * (len(labels) - 1) / 2, categories)
plt.legend()

plt.tight_layout()
plt.show()
