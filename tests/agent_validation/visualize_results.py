import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    """Загружает данные результатов валидации из указанного JSON файла."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(data):
    """Рассчитывает средние значения score и tool_score для каждой модели."""
    metrics = {}
    for model, results in data.items():
        scores = [item.get('score', 0) for item in results]
        tool_scores = [item.get('tool_score', 0) for item in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_tool_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0
        metrics[model] = {'avg_score': avg_score, 'avg_tool_score': avg_tool_score}
    return metrics

def create_visualization(metrics, output_path):
    """Формирует и сохраняет графическое представление метрик моделей."""
    models = list(metrics.keys())
    avg_scores = [metrics[m]['avg_score'] for m in models]
    avg_tool_scores = [metrics[m]['avg_tool_score'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_scores, width, label='Score (0-5)')
    ax.bar(x + width/2, avg_tool_scores, width, label='Tool Score (0-1)')

    ax.set_ylabel('Оценка')
    ax.set_title('Результаты валидации агентов')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Основная точка входа для генерации визуализации результатов."""
    data = load_data('/home/uwi/my_workspace/2026/testovoe_ingosstrah/agent_validation/validation_results.json')
    metrics = calculate_metrics(data)
    create_visualization(metrics, '/home/uwi/my_workspace/2026/testovoe_ingosstrah/agent_validation/validation_chart.png')

if __name__ == '__main__':
    main()
