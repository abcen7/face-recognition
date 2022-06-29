import sys
import os
import cv2
import pickle
import sqlite3
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from create_yml_from_video import create_yml


class FilterDial(QDialog):
    def __init__(self):
        super(FilterDial, self).__init__()
        uic.loadUi('design/filters.ui', self)
        self.setWindowIcon(QIcon("photos/Icon.png"))
        self.accept_btn.setIcon(QIcon('photos/check.png'))
        self.accept_btn.clicked.connect(self.accept)
        self.slider_blur.setEnabled(False)
        self.slider_sharpen.setEnabled(False)
        self.slider_blur.valueChanged.connect(self.lcdNumber_blur.display)
        self.slider_sharpen.valueChanged.connect(self.lcdNumber_sharpen.display)
        self.blur.stateChanged.connect(self.turn_blur)
        self.monochrome.stateChanged.connect(self._turn_off)
        self.sharpen.stateChanged.connect(self.turn_sharpen)

    def _turn_off(self):
        boxes = [self.blur, self.monochrome, self.sharpen]
        uncheckeds = [box for box in boxes if not box.isChecked()]
        for box in uncheckeds:
            box.setDisabled(len(uncheckeds) == 2)

    def turn_blur(self):
        self._turn_off()
        self.slider_blur.setEnabled(self.blur.isChecked())

    def turn_sharpen(self):
        self._turn_off()
        self.slider_sharpen.setEnabled(self.sharpen.isChecked())

    def get_data(self):
        filters = [el for el in [self.blur, self.monochrome, self.sharpen] if el.isChecked()]
        if filters:
            if filters[0] == self.sharpen:
                return filters[0].text().strip(), self.slider_sharpen.value() / 10
            if filters[0] == self.blur:
                return filters[0].text().strip(), self.slider_blur.value() / 10
            if filters[0] == self.monochrome:
                return filters[0].text().strip(), None
        return None, None


class CameraCv(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent):
        super(CameraCv, self).__init__()
        self.create = False
        self.stop = False  # Флаг остановки видеопотока
        if not os.path.exists(r"data/recognition.yml"):
            self.stop = True
        if not os.path.exists(r"data/labels.pickle"):
            self.stop = True
        self.blur = False  # Флаг размытия изображения
        self.degree_blur = None  # Степень размытия
        self.monochrome = False  # Флаг чб
        self.sharpen = False  # Флаг контраста изображения
        self.degree_sharpen = None  # Степень контраста

    def face_id(self, recognizer, labels, grey_frame, color_frame, x, y, w, h):
        roi_gray = grey_frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        if conf <= 70:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Log in"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(color_frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "No name"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(color_frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(color_frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        return color_frame

    def do_sharpen(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Converting image to LAB Color model
        l, a, b = cv2.split(lab)  # Splitting the LAB image to different channels
        clahe = cv2.createCLAHE(clipLimit=max(1, self.degree_sharpen),
                                tileGridSize=(7, 7))  # Applying CLAHE to L-channel
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))  # Merge the CLAHE enhanced L-channel with the a and b channel
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # Converting image from LAB Color model to RGB model

    def do_blur(self, frame):
        print(self.degree_blur)
        return cv2.GaussianBlur(frame, (7, 7), max(1, self.degree_blur))

    def do_monochrome(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def run(self):
        """Здесь будет проискходить работа с камерой
        (получение видое, распознавание лиц, накладывание фильтров)"""
        face_cascade = cv2.CascadeClassifier('data/head_cascade.xml')
        # eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
        # smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
        if os.path.exists(r"data/recognition.yml"):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(r"data/recognition.yml")
        else:
            self.create = True
        if os.path.exists(r"data/labels.pickle"):
            with open("data/labels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {v: k for k, v in og_labels.items()}
        else:
            self.create = True
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret and not self.stop:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
                if self.create:
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(r"data/recognition.yml")
                    with open("data/labels.pickle", 'rb') as f:
                        og_labels = pickle.load(f)
                        labels = {v: k for k, v in og_labels.items()}
                    self.create = False
                for (x, y, w, h) in faces:
                    frame = self.face_id(recognizer, labels, gray, frame, x, y, w, h)
                if self.sharpen:
                    frame = self.do_sharpen(frame)
                elif self.blur:
                    frame = self.do_blur(frame)
                elif self.monochrome:
                    frame = self.do_monochrome(frame)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_qt.scaled(640, 480, Qt.KeepAspectRatio)  # Здесь задоются размеры окна видео
                self.changePixmap.emit(p)

    def stop_show(self):
        self.stop = True  # Выключаем видео

    def continue_show(self):
        self.stop = False
        if not os.path.exists(r"data/recognition.yml"):
            self.stop = True
        if not os.path.exists(r"data/labels.pickle"):
            self.stop = True

    def set_data(self, name, coeff):
        self.stop = self.blur = self.monochrome = self.sharpen = False
        self.degree_blur = self.degree_sharpen = None
        if name == "Sharpen":
            self.sharpen = True
            self.degree_sharpen = coeff
        elif name == "Blur":
            self.blur = True
            self.degree_blur = coeff
        elif name == "Monochrome":
            self.monochrome = True

    def set_create(self, status):
        self.create = status


class BDDial(QDialog):
    def __init__(self):
        super(BDDial, self).__init__()
        self.setWindowIcon(QIcon("photos/Icon.png"))
        uic.loadUi('design/bd_people.ui', self)
        self.db = DataBase()
        self.create = False
        self.download_btn.setIcon(QIcon('photos/video.png'))

        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 300)
        self.show_table()

        self.add_to_bd_button.clicked.connect(self.add_person)
        self.download_btn.clicked.connect(self.select_video)

    def show_table(self):
        self.create = False
        data = self.db.get_niknames()
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Имя", "Фамилия"])
        self.tableWidget.setRowCount(len(data))
        for i in range(len(data)):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(data[i].split('_')[0]))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(data[i].split('_')[1]))
        self.tableWidget.resizeColumnsToContents()

    def add_person(self):
        if self.name_le.text().strip() != '' and self.le_surname.text().strip() != '' \
                and self.video_lbl.text().strip() != '':
            self.create = True
            self.db.add_nikname(self.name_le.text(), self.le_surname.text())
            data = [(el, None) for el in self.db.get_niknames()[:-1]]
            data.append((self.db.get_niknames()[-1], self.video_lbl.text()))
            create_yml(data)
            self.show_table()
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка")
            msg.setText("Вы оставили какое то поле пустым")
            msg.setIcon(QMessageBox.Critical)
            msg.exec_()

    def select_video(self):
        self.create = False
        f_name, _ = QFileDialog.getOpenFileName(
            self, 'Select video', '',
            'Video (*.mp4 *.mov);;Все файлы (*)')
        self.video_lbl.setText(f_name)

    def get_create(self):
        return self.create


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('design/camera.ui', self)
        self.setWindowIcon(QIcon("photos/Icon.png"))
        self.camera.setIcon(QIcon('photos/pngwing.com.png'))
        self.filters.setIcon(QIcon('photos/filters.png'))
        self.bd_of_people.setIcon(QIcon('photos/audience (1).png'))
        # Создание подключений
        self.filters.clicked.connect(self.open_filter)
        self.bd_of_people.clicked.connect(self.open_bd)
        # Подключение видеопотока
        self.camera = CameraCv(self)
        self.camera.changePixmap.connect(self.setImage)
        self.camera.start()

    def setImage(self, image):
        self.opencv_label.setPixmap(QPixmap.fromImage(image))

    def open_filter(self):
        self.camera.stop_show()
        dlg_filter = FilterDial()
        if dlg_filter.exec_():
            print(dlg_filter.get_data())
            self.camera.set_data(*dlg_filter.get_data())
        self.camera.continue_show()

    def open_bd(self):
        self.camera.stop_show()
        dlg_bd = BDDial()
        if not dlg_bd.exec_():
            print(dlg_bd.get_create())
            self.camera.set_create(dlg_bd.get_create())
        self.camera.continue_show()


class DataBase:
    def __init__(self):
        self.db = sqlite3.connect(r"data/persons.SQLITE")

    def get_niknames(self):
        cur = self.db.cursor()
        data = cur.execute("SELECT nikname FROM persons").fetchall()
        cur.close()
        return [el[0] for el in data]

    def add_nikname(self, name, surname):
        cur = self.db.cursor()
        data = name + '_' + surname
        cur.execute("INSERT INTO  persons (nikname) VALUES (?)", (data,))
        self.db.commit()
        cur.close()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())
