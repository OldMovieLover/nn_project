import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Функция для визуализации логов
def load_and_visualize_log(log_file):
    # Загрузка данных из CSV
    df = pd.read_csv(log_file)
    
    # Отображение таблицы с метриками
    st.write("Логи обучения:")
    st.dataframe(df)
    
    # Построение графиков кривых потерь и точности
    plt.style.use("seaborn-v0_8-talk")
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # График потерь
    ax[0].plot(df['Epoch'], df['Train_Loss'], label='Train_Loss')
    ax[0].plot(df['Epoch'], df['Val_Loss'], label='Validation_Loss')
    ax[0].set_title('Кривая потерь')
    ax[0].set_xlabel('Эпоха')
    ax[0].set_ylabel('Потери')
    ax[0].legend()

    # График точности
    ax[1].plot(df['Epoch'], df['Train_Accuracy'], label='Train_Accuracy')
    ax[1].plot(df['Epoch'], df['Val_Accuracy'], label='Validation_Accuracy')
    ax[1].set_title('Кривая точности')
    ax[1].set_xlabel('Эпоха')
    ax[1].set_ylabel('Точность')
    ax[1].legend()

    st.pyplot(fig)

    # F1-метрика
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    ax_f1.plot(df['Epoch'], df['train_f1'], label='Train F1')
    ax_f1.plot(df['Epoch'], df['val_f1'], label='Val F1')
    ax_f1.set_title('Кривая F1-метрики')
    ax_f1.set_xlabel('Эпоха')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.legend()

    st.pyplot(fig_f1)

# Если выбрана главная страница

st.header("Сводная информация по моделям")
    
# Загрузка и отображение логов для природных условий
st.subheader("Сводная информация по модели 'природные условия'")
st.write("Время обучения нейронной сети: 87.910699 секунд")
load_and_visualize_log('logs/log_file')

# Загрузка и отображение логов для птичек
st.subheader("Сводная информация по модели 'птички'")
st.write("Время обучения нейронной сети: 488.213757 секунд")
load_and_visualize_log('logs/log_file_dasnet')

