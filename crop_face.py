import cv2
import os

def Cutting_face_save(image, save_dir, num):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i, (x, y, w, h) in enumerate(faces):
        cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (180, 180))

        # 저장할 디렉토리가 없으면 생성
        os.makedirs(save_dir, exist_ok=True)

        # 이미지 저장하기
        save_path = os.path.join(save_dir, f"{num}.jpg")
        cv2.imwrite(save_path, resize)
        print(f"Saved: {save_path}")


# 이미지 파일 경로
# path_dir = "./download/"
# path_list = os.listdir(path_dir)
path_list = [21]
for name in path_list:
    path_d =  f"./download/{name}/"
    file_list = os.listdir(path_d)
    # 각 이미지에 대해 얼굴을 잘라서 저장
    for num,file_name in enumerate(file_list):
        image = cv2.imread(os.path.join(path_d, file_name))
        Cutting_face_save(image, f"img/{name}/", num)
