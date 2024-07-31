import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, send_file, render_template_string
import os

app = Flask(__name__)

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

# Сохранение графика в файл
output_image = 'results_plot.png'
plt.savefig(output_image)
plt.close(fig)


@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Training and Inference Time Comparison</title>
    </head>
    <body>
        <h1>Training and Inference Time Comparison</h1>
        <img src="/plot.png" alt="Training and Inference Time Comparison">
    </body>
    </html>
    ''')


@app.route('/plot.png')
def plot_png():
    return send_file(output_image, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
