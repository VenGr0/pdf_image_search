import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Загрузка и разбор PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Разделение текста на предложения
def split_into_sentences(text):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

# Поиск релевантных фрагментов текста
def search_relevant_sentences(query, sentences, top_k=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences + [query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = np.argsort(cosine_similarities[0])[-top_k:][::-1]
    return [sentences[i] for i in top_indices]

# Загрузка модели CLIP
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Векторизация текста с помощью CLIP
def vectorize_text(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.numpy().flatten()

# Векторизация изображения с помощью CLIP
def vectorize_image(image, model, processor):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.numpy().flatten()

# Поиск схожих изображений
def find_similar_images(query_vector, image_vectors, top_k=3):
    index = faiss.IndexFlatL2(query_vector.shape[0])
    index.add(np.array(image_vectors))
    distances, indices = index.search(np.array([query_vector]), top_k)
    return indices[0]

# Основная функция
def main():
    # Путь к PDF-документу
    pdf_path = "1984.pdf"  # Укажите путь к вашему PDF
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    
    # Поиск по тексту
    query = input("Введите поисковый запрос: ")
    relevant_sentences = search_relevant_sentences(query, sentences)
    
    print("Топ-3 релевантных фрагмента:")
    for i, sentence in enumerate(relevant_sentences, 1):
        print(f"{i}. {sentence}")
    
    # Загрузка датасета CIFAR-10
    cifar10 = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    
    # Загрузка модели CLIP
    model, processor = load_clip()
    
    # Векторизация изображений из CIFAR-10
    image_vectors = []
    images = []
    for idx in range(len(cifar10)):
        image, _ = cifar10[idx]
        image = image.permute(1, 2, 0).numpy()  # Преобразуем в формат HWC
        image = Image.fromarray((image * 255).astype(np.uint8))  # Преобразуем в PIL Image
        vector = vectorize_image(image, model, processor)
        image_vectors.append(vector)
        images.append(image)
        if idx >= 999:  # Ограничимся 1000 изображениями для скорости
            break
    
    # Векторизация текстового запроса с помощью CLIP
    query_vector = vectorize_text(query, model, processor)
    
    # Поиск схожих изображений
    similar_image_indices = find_similar_images(query_vector, image_vectors)
    
    # Вывод схожих изображений
    print("\nТоп-3 схожих изображения из датасета CIFAR-10:")
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(similar_image_indices, 1):
        plt.subplot(1, 3, i)
        plt.imshow(images[idx])
        plt.title(f"Изображение {i}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()