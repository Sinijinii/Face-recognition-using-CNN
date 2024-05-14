import os
import csv

# 이미지 파일이 있는 디렉토리 경로
image_dir = "./img/"
name_list = list(map(str,range(1,31)))


# image_dir = "./img/"
# name_list = os.listdir(image_dir)

# list_num = 0
# print(name_list)
# for i in name_list:
#     list_num += len(os.listdir(f"./img/{i}"))
# print(list_num)
# CSV 파일 경로
csv_file_path = "./labels.csv"

# 파일 이름을 라벨로 사용하여 CSV 파일에 데이터 추가
with open(csv_file_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Label"])

    # 각 파일에 대해 이미지 수를 구하고, 해당 이미지 수만큼 파일 이름을 라벨로 사용하여 CSV 파일에 데이터 추가
    for file_name in name_list:
        file_path = os.path.join(image_dir, file_name)
        image_count = len(os.listdir(file_path))
        for _ in range(image_count):
            writer.writerow([file_name])

print("CSV 파일이 생성되었습니다.")
