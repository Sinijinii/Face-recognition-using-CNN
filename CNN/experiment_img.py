from preprocess import *
from CNN_MODEL import DeepLearning

from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import save_model
from config import *


dl = DeepLearning()


def experiment_2d_cnn(X_tr, y_tr, X_val, y_val, X_te, y_te, result, exp_args, vid_type):
    print('2D CNN')

    # normalize
    X_tr, X_val, X_te = normalize(X_tr), normalize(X_val), normalize(X_te)
    print('---------------experiment_2d_cnn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-----------------')
    print(X_tr)
    # build neural network
    #img_height, img_width, img_ch = exp_args[vid_type]['rgb']['height'], exp_args[vid_type]['rgb']['width'], X_tr.shape[3]
    img_height, img_width, img_ch = X_tr.shape[1], X_tr.shape[2], X_tr.shape[3]
    model = dl.build_cnn_2d(img_height = img_height, img_width = img_width, img_ch = img_ch)

    # run neural network
    model, history  = dl.run_model(model = model, X_tr=X_tr, y_tr = y_tr, X_val = X_val, y_val = y_val)
    te_loss, te_acc = model.evaluate(X_te, y_te, batch_size = dl.batch_size)

    # store dataset
    result['X_tr'], result['X_val'], result['X_te'] = X_tr, X_val, X_te
    result['y_tr'], result['y_val'], result['y_te'] = y_tr, y_val, y_te

    # store loss, accuracy
    result['tr_loss_list'], result['val_loss_list'], result['te_loss'] = history.history['loss'],     history.history['val_loss'],     te_loss
    result['tr_acc_list'],  result['val_acc_list'],  result['te_acc']  = history.history['accuracy'], history.history['val_accuracy'], te_acc

    return model, result
