import cv2
import numpy as np
import os
from PIL import Image
import pickle


def create_images_for_person(person, video_dir):
    """Функция которая нарежет черно белые кадры из видое
    person - имя_фамилия человека"""
    # Поддержтваемые форматы видео:
    # 3gp, avi, f4v, hevc, mkv, mov, mp4, mpg, ts, webm, wmv

    SAVING_FRAMES_PER_SECOND = 5  # Интервал выборки кадров 17 если видео минута
    SAVING_PATH = f'images/{person}'
    if not os.path.exists(SAVING_PATH):
        os.mkdir(SAVING_PATH)
    cap = cv2.VideoCapture(video_dir)  # Открываем видео на обработку
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS исходного видео
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)  # Выбираем меньшее значение FPS:

    # вдруг FPS исходного видео меньше чем задано для выборки
    frames_timecodes = []  # Получаем таймкоды на нужные кадры
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)  # Длительность клипа
    for i in np.arange(0, clip_duration, 1 / saving_frames_per_second):
        frames_timecodes.append(i)
    cur_frame = 0  # Счетчик для всех кадров
    img_count = 1  # Счетчик для кадров, которые впоследствие будут отобраны
    while img_count <= 400:  # Запускаем цикл выборки кадров
        print(img_count)
        is_read, frame = cap.read()  # Считываем кадр
        if not is_read:  # Если кадров больше не осталось, выходим из цикла
            break
        frame_duration = cur_frame / fps  # Вычисляем таймкод текущего кадра
        try:
            closest_duration = frames_timecodes[img_count - 1]  # Берем  таймкод из массива нужных таймкодов
        except IndexError:
            break  # Если массив закончился, то срабатывает исключение, т.е. все нужные кадры записаны
        # Если таймкод текущего кадра больше или равен чем таймкод из массива, значит сохраняем кадр
        if frame_duration >= closest_duration:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Переводим изображение в черно-белую палитру
            cv2.imwrite(os.path.join(SAVING_PATH, f"{img_count}.jpg"), gray)  # Записываем кадр в папку 'frames'
            img_count += 1  # Увеличиваем счетчик сохраненных кадров
        cur_frame += 1  # Увеличиваем счетчик всех пройденных кадров


def forming_data_source():
    if os.path.exists(r"data/recognition.yml"):
        os.remove(r"data/recognition.yml")
    if os.path.exists(r"data/labels.pickle"):
        os.remove(r"data/labels.pickle")
    face_cascade = cv2.CascadeClassifier('data/head_cascade.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []
    image_dir = "images/"
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(label)
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                size = (550, 550)
                final_image = Image.open(path).resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("data/labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("data/recognition.yml")


def create_yml(people):
    for name, video_dir in people:
        if not os.path.exists(f"images/{name}"):
            create_images_for_person(name, video_dir)
    print("Learning")
    forming_data_source()


if __name__ == '__main__':
    create_yml([("Andrey_Kizhinov", r"D:\Videos\20211204_153328.mp4"),
                ("Alexander_Gorobchenko", r"D:\Videos\20211204_153217.mp4"),
                ("Natasha_Trofimova", r"D:\Programing\Python\face_detect2.0\Images\natasha\IMG_0368.mov"),
                ("Kirill_Sermyagin", r"D:\Videos\20211204_155029.mp4")])
