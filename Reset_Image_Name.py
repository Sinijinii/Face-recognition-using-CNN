import os

# 이미지 파일이 있는 디렉토리 경로
img = list(map(str,range(1,31)))
for img_name in img:
    
    image_dir = f"C:/Users/SSAFY/Desktop/시니지니/CNN이미지/Face-recognition-using-CNN/img/{img_name}/"

    # 파일 번호 초기화
    file_num = 1

    # 파일을 순서대로 읽어서 파일 이름 변경
    for file_name in sorted(os.listdir(image_dir)):
        # 새로운 파일 이름 설정
        new_file_name = f"{file_num:03d}.jpg"
        # 파일의 이전 이름과 새 이름을 출력
        print(f"Renaming {file_name} to {new_file_name}")
        # 파일 이름 변경
        os.rename(os.path.join(image_dir, file_name), os.path.join(image_dir, new_file_name))
        # 파일 번호 증가
        file_num += 1

    print("파일 이름 변경이 완료되었습니다.")
