from sympy import false
from CNN_MODEL import DeepLearning
import numpy as np
from preprocess import *
from experiment_img import *
from preprocess import *
from result_csv import *


dl = DeepLearning()

##################################################################################################################################################################
# define functions
##################################################################################################################################################################
# experiment from image data

print("experiment rgb")

train_folder = 'train_img'
test_folder = 'test_img'
labels_file = 'labels.csv'
print('--------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!___________________________')
model, result = experiment_2d_cnn(train_folder, test_folder, labels_file, {})


# result dataframe
#res_df_rgb = create_result()

# print(f"test set len: {len(result_rgb['X_te'])} / rgb val acc: {round(result_rgb['val_acc_list'][-1], 4)} / te acc: {round(result_rgb['te_acc'], 4)}")
