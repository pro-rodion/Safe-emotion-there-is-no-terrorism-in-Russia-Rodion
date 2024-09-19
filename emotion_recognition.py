import cv2
import numpy as np
from deepface import DeepFace
from scipy.signal import find_peaks
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import filedialog

emotion = "Неизвестно"

# Словарь перевода эмоций
emotion_translations = {
    "angry": "Злой",
    "disgust": "Отвращение",
    "fear": "Страх",
    "happy": "Счастливый",
    "sad": "Грустный",
    "surprise": "Удивление",
    "neutral": "Нейтральный"
}

def calculate_pulse(roi, pulse_history):
    if roi is not None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        mean_brightness = np.mean(value_channel)
        pulse_history.append(mean_brightness)

        if len(pulse_history) > 100:
            pulse_history.pop(0)

        return pulse_history
    return pulse_history

def put_text(img, text, position, color=(255, 255, 255)):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)

    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)


def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > height:
        new_width = min(max_width, width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(max_height, height)
        new_width = int(new_height * aspect_ratio)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def recognize_emotions_and_pulse(frame_source):
    global emotion
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    pulse_history = []
    emotion_count = defaultdict(int)

    cap = cv2.VideoCapture(frame_source)
    if frame_source == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(frame_source)
    # Установка размеров окна
    cv2.namedWindow('Распознавание эмоций и пульса', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        # Если файл закончился, сбросить индекс видео
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Сбросить видеопоток на начало
            continue

        # Изменение размера изображения с учетом пропорций
        frame = resize_image(frame, 800, 600)
        cv2.resizeWindow('Распознавание эмоций и пульса', frame.shape[1], frame.shape[0])

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_face = frame[y:y + h, x:x + w]

            pulse_history = calculate_pulse(roi_face, pulse_history)

            try:
                analysis = DeepFace.analyze(roi_face, actions=['emotion'], enforce_detection=True)
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion = emotion_translations.get(dominant_emotion, "Неизвестно")
                emotion_count[emotion] += 1
            except Exception:
                emotion = "Неизвестно"

            if len(pulse_history) >= 10:
                peaks, _ = find_peaks(pulse_history, distance=5)
                pulse_rate = (len(peaks) / (len(pulse_history) / 60)) * 60
                corrected_pulse = pulse_rate * 0.75 / 5.75
                frame = put_text(frame, f'Пульс: {int(corrected_pulse)} уд./мин', (x, y - 60), color=(0, 255, 0))
            else:
                frame = put_text(frame, 'Пульс: Не обнаружен', (x, y - 60))

            frame = put_text(frame, f'Текущая эмоция: {emotion}', (x, y - 30))

            if emotion_count:
                most_frequent_emotion = max(emotion_count, key=emotion_count.get)
                frame = put_text(frame, f'Частая эмоция: {most_frequent_emotion}', (x, y + h + 10),
                                 color=(0, 0, 255))

        cv2.imshow('Распознавание эмоций и пульса', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def start_recognition_camera():
    root.destroy()  # Закрыть главное окно
    recognize_emotions_and_pulse('camera')


def start_recognition_file():
    file_path = filedialog.askopenfilename(title="Выберите изображение или видео",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                                                      ("Video files", "*.mp4 *.avi")])
    if file_path:
        root.destroy()  # Закрыть главное окно
        recognize_emotions_and_pulse(file_path)


def show_main_menu():
    global root, background_photo
    root = tk.Tk()
    root.title("Главное меню")

    # Загрузка изображения для фона
    background_image = Image.open("project.jpg")  # Используйте PIL Image
    background_photo = ImageTk.PhotoImage(background_image)

    # Настройка фона
    background_label = tk.Label(root, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Кнопка для начала распознавания с камеры
    camera_button = tk.Button(root, text="Сканирование с камеры", command=start_recognition_camera)
    camera_button.pack(pady=250)

    # Кнопка для начала распознавания с файла
    file_button = tk.Button(root, text="Сканирование из файла", command=start_recognition_file)
    file_button.pack(pady=1)

    root.geometry("720x720")
    root.mainloop()


if __name__ == '__main__':
    show_main_menu()