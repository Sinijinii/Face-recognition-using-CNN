from pathlib import Path
import sys
import numpy as np
import pandas as pd

def create_result(result_path, result_file_name, test_target_dict, remove_target_idx, exp_idx, data_type, model, result, save_csv):
    # set file directory and name
    # f'../result/real/{SUBJECT}/uni_modal/' + f'res_uni_modal_{DATA_VERSION}_{MODEL_VERSION}.csv',
    result_file_name = result_path['uni_modal'] + result_file_name['uni_modal'] + '_' + data_type + '.csv'

    # extract list of test set class
    te_class_list = [list(test_target_dict.values())[i-1] for i in test_target_dict.keys() if i not in remove_target_idx]

    # initialize result data frame
    columns    = ['exp_idx'] + te_class_list + ['pred_class', 'real_class', 'tr_acc']
    result_df  = pd.DataFrame(columns = columns)

    # extract class/prediction
    real_class   = result['y_te'].argmax(axis = -1) + 1  # real class
    result_prob  = model.predict(result['X_te'])        # predicted probability
    result_class = result_prob.argmax(axis = -1) + 1    # predicted class

    # assign values in result data frame
    num_rows = len(result_prob)
    for i in range(num_rows):
        result_df.loc[i, 'exp_idx']     = exp_idx
        result_df.loc[i, te_class_list] = result_prob[i]
        result_df.loc[i, 'pred_class']  = result_class[i]
        result_df.loc[i, 'real_class']  = real_class[i]
        result_df.loc[i, 'tr_acc']      = result['tr_acc_list'][-1]
    # end for

    if save_csv:
        # save result
        file = Path(result_file_name)
        if not file.is_file():
            result_df.to_csv(result_file_name, index = True)
        else:
            result_df.to_csv(result_file_name, mode = 'a', header = False, index = True)
        # end if
    # end if

    return result_df