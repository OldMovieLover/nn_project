import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели для определения природных явлений
model_recognition = models.resnet18(pretrained=False)
model_recognition.fc = torch.nn.Linear(model_recognition.fc.in_features, 11) 
model_recognition.load_state_dict(torch.load('models/model_resnet18.pth', map_location=device))
model_recognition.eval()

# Список названий классов
class_names_recognition = [
    "роса", "туман, смог", "иней", "глазурь", "град", "молния", "дождь",
    "радуга", "изморозь", "песчаная буря", "снег"
]

# Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция для предсказания
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model_recognition(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Функция для загрузки изображения по URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Интерфейс для страницы
st.header("Определение природного явления по картинке")
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
            prediction = predict(img)
            predicted_class = class_names_recognition[prediction]
            st.write(f"Природное явление {i+1}: {predicted_class}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
    elif url:
        start_time = time.time()
        prediction = predict(image)
        predicted_class = class_names_recognition[prediction]
        st.write(f"Предсказание: {predicted_class}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
    else:
        st.warning("Пожалуйста, загрузите изображение.")