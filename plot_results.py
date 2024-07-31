import matplotlib.pyplot as plt

# Чтение результатов
results_cpu_pytorch = open('results_cpu_pytorch.txt').read().split(',')
results_gpu_pytorch = open('results_gpu_pytorch.txt').read().split(',')
results_cpu_crypten = open('results_cpu_crypten.txt').read().split(',')
results_gpu_crypten = open('results_gpu_crypten.txt').read().split(',')

methods = ['CPU PyTorch', 'GPU PyTorch', 'CPU CrypTen', 'GPU CrypTen']
training_times = [float(results_cpu_pytorch[0]), float(results_gpu_pytorch[0]), float(results_cpu_crypten[0]), float(results_gpu_crypten[0])]
inference_times = [float(results_cpu_pytorch[1]), float(results_gpu_pytorch[1]), float(results_cpu_crypten[1]), float(results_gpu_crypten[1])]

# График времени обучения
plt.figure(figsize=(10, 5))
plt.bar(methods, training_times, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Method')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.show()

# График времени инференса
plt.figure(figsize=(10, 5))
plt.bar(methods, inference_times, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Method')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time Comparison')
plt.show()
