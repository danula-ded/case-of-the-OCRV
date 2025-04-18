import pandas as pd
import matplotlib.pyplot as plt

def getTrainInfo():
    # Загрузка данных
    df = pd.read_csv('./data/train_1_5000.csv')

    # Сохранение отчета в файл
    with open('data_report.txt', 'w', encoding='utf-8') as f:
        # 1. Основная информация о данных
        f.write("===================\n")
        f.write("Основная информация:\n")
        f.write("===================\n")
        df.info(buf=f)
    
        # 2. Описательные статистики
        f.write("\n\n===================\n")
        f.write("Описательные статистики:\n")
        f.write("===================\n")
        f.write(str(df.describe().T))
    
        # 3. Количество уникальных значений
        f.write("\n\n===================\n")
        f.write("Уникальные значения:\n")
        f.write("===================\n")
        f.write(str(df.nunique()))
    
        # 4. Пропущенные значения
        f.write("\n\n===================\n")
        f.write("Пропущенные значения:\n")
        f.write("===================\n")
        f.write(str(df.isnull().sum()))
    
        # 5. Распределение целевой переменной (incident)
        if 'incident' in df.columns:
            f.write("\n\n===================\n")
            f.write("Распределение целевой переменной:\n")
            f.write("===================\n")
            f.write(str(df['incident'].value_counts()))

    # 6. Визуализация нескольких признаков
    num_plots = 5  # Количество признаков для визуализации
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.columns[:num_plots]):
        plt.subplot(2, 3, i+1)
        df[col].hist(bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.savefig('sample_distributions.png')
    plt.show()

    print("Анализ завершен! Проверьте файлы:")
    print("- data_report.txt - текстовый отчет")
    print("- sample_distributions.png - примеры распределений")


getTrainInfo()