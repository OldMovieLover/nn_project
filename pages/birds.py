import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import time

# Загрузка модели для определения птичек
model_birds = models.densenet121(pretrained=False)
model_birds.classifier = torch.nn.Linear(model_birds.classifier.in_features, 200)
checkpoint = torch.load('models/model_desnet121.pth')
model_birds.load_state_dict(checkpoint, strict=False)
model_birds.eval()

# Загрузка названий птичек
with open('models/birds_name.txt', 'r') as file:
    data = file.read()
class_names_bird = eval(data)

# Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция для предсказания
def predict_birds(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model_birds(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Функция для загрузки изображения по URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Интерфейс для страницы
st.header("Определение вида птички по картинке")
image_option = st.radio(
    "Выберите способ загрузки изображения:",
    ("Загрузить изображение из файла", "Загрузить изображение по URL")
)

if image_option == "Загрузить изображение из файла":
    uploaded_files = st.file_uploader("Выберите изображение...", type="jpg", accept_multiple_files=True)
    if uploaded_files:
        images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
        for i, img in enumerate(images):
            st.image(img, caption=f"Загруженное изображение {i+1}", use_column_width=True)

elif image_option == "Загрузить изображение по URL":
    url = st.text_input("Введите URL изображения:")
    if url:
        uploaded_files = None
        try:
            image = load_image_from_url(url)
            st.image(image, caption="Изображение из URL", use_column_width=True)
        except Exception as e:
            st.error(f"Не удалось загрузить изображение. Ошибка: {e}")

if st.button("Предсказать"):
    if uploaded_files:
        start_time = time.time()
        for i, img in enumerate(images):
            prediction = predict_birds(img)
            predicted_class = class_names_bird[prediction]
            st.write(f"Птичка {i+1}: {predicted_class}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
    elif url:
        start_time = time.time()
        prediction = predict_birds(image)
        predicted_class = class_names_bird[prediction]
        st.write(f"Предсказание: {predicted_class}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
    else:
        st.warning("Пожалуйста, загрузите изображение.")