import cv2
import os
import numpy as np

# 데이터 불러오기 및 전처리
def load_data(train_folder, test_folder, labels_file):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # train 이미지 불러오기
    for i in range(1, 31):  # 1부터 30까지의 폴더
        folder_path = os.path.join(train_folder, str(i))
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            # 이미지를 읽어와서 numpy 배열로 변환
            image = cv2.imread(image_path)
            image = cv2.resize(image, (35, 35))
            train_images.append(image)
            # 라벨 추가
            train_labels.append(i)

    # test 이미지 불러오기
    for i in range(1, 31):  # 1부터 30까지의 폴더
        folder_path = os.path.join(test_folder, str(i))
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            # 이미지를 읽어와서 numpy 배열로 변환
            image = cv2.imread(image_path)
            image = cv2.resize(image, (35, 35))
            test_images.append(image)
            # 라벨 추가
            test_labels.append(i)

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

# 데이터 경로 설정
# train_folder = 'train_img'
# test_folder = 'test_img'
# labels_file = 'labels.csv'

# # 데이터 불러오기
# train_images, train_labels, test_images, test_labels = load_data(train_folder, test_folder, labels_file)
# print(train_images)



def standardize(X_tr, X_te):
	"""
	:param X_tr: training set to be standardized
	:param X_te: test set to be standardized
	:return X_tr_norm: normalized training set
	:return X_te_norm: normalized test set
	"""
	print('Standardize dataset')

	# set dimension of input
	dim = len(X_tr.shape)

	if dim == 2: # 2D
		X_tr_mean = np.mean(X_tr, axis = 0)[None, :]
		X_tr_std  = np.std(X_tr, axis = 0)[None, :]

		X_tr_norm = (X_tr - X_tr_mean) / X_tr_std
		X_te_norm = (X_te - X_tr_mean) / X_tr_std
	else: # 3D
		X_tr_mean = np.mean(X_tr, axis = 0)[None, :, :]
		X_tr_std  = np.std(X_tr, axis = 0)[None, :, :]

		X_tr_norm = (X_tr - X_tr_mean) / X_tr_std
		X_te_norm = (X_te - X_tr_mean) / X_tr_std
	# end if

	return X_tr_norm, X_te_norm




# perform one-hot encoding
def one_hot_encoding(y, n_class):
	"""
	:param y: target vector that should be encoded
	:param n_class: number of classes in target
	:return encoded_target: target vector after one-hot encoding
	"""
	print('One-hot encode target')

	# n x m identity matrix
	expansion = np.eye(n_class)
	y         = y.astype(int)
	encoded_y = expansion[:, y - 1].T

	#assert y.shape[1] == n_class, "Wrong number of labels!"

	return encoded_y
