
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models_Check_parameter2 import ResearchModels
import time
import os.path
import os
from keras.utils import to_categorical
import scipy.io
import numpy as np

from keras import backend as K
import glob
from os.path import splitext, basename
import numpy as np
import matplotlib.pylab as plt

from keras.models import load_model

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

# # all_band, gamma, beta, alpha, theta, delta
# band = 'all_band'
#
# # First, Second, Third
# group = 'Second_BI'
#
# # window = '0.636'



dir_rat19 = os.path.join(os.getcwd(),'rat_newstim19_all.mat')
# file_list = os.listdir(os.getcwd())
# file_list.sort()
#
# for filename in file_list:
#     print('filename : ', filename)
#     file_base = splitext(basename(filename))[0]
#     print(file_base)
#
# for filename in glob.glob('/home/bravo/workspace/YHJ/JS_Data/final_two_ch/'+ band + '/' + group + '/*.mat') :
#     print(filename)

# model saved folder path (lstm_model or bi_lstm_model)

MODEL_SAVE_FOLDER_PATH = './bi_lstm_model_pm2/'

for filename in glob.glob(dir_rat19):
    print('filename : ', filename)
    file_base = splitext(basename(filename))[0]

    K.clear_session()

    model = 'bi_lstm'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 151
    batch_size = 64
    nb_epoch = 1000

    # Helper: Stop when we stop learning.
    # validation set이 없으므로 early stopping은 제외해야 할듯
    # early_stopper = EarlyStopping(patience=nb_epoch)

    mat = scipy.io.loadmat(filename)

    y = mat['label_set']
    X = np.array(mat['data_set'])
    X = np.reshape(X, [-1, 151, 2])

    # batch_size = X.shape[0]

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)

    fold_no = 1

    for train_index, test_index in skf.split(X, y):
        print('------------------------------------')
        print('Training for fold {}.'.format(fold_no))

        y = to_categorical(mat['label_set'], 5)  # one-hot encoding # categorical(y, 5)로 하면 error
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        LSTM = ResearchModels(5, model, seq_length, saved_model, features_length=2)
        print('*****', file_base, '*****')

        model_path = MODEL_SAVE_FOLDER_PATH + file_base + "-" + "fold" + str(fold_no) + '.hdf5'

        #         model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
        #         model_path = MODEL_SAVE_FOLDER_PATH + file_base + "-" + "fold" + str(fold_no) + "-" + '{epoch:02d}-{val_loss:.4f}.hdf5'

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1,
                                        save_best_only=True, mode="max")

        hist = LSTM.model.fit(X_train, y_train, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test),
                              epochs=nb_epoch, callbacks=[cb_checkpoint])

        # training acc and loss + validation acc and loss save

        result_training_acc = hist.history['acc']
        result_training_loss = hist.history['loss']
        result_validation_acc = hist.history['val_acc']
        result_validation_loss = hist.history['val_loss']

        #         print('\n')
        #         print('## training loss and acc ##')
        #         print('training loss : ', hist.history['loss'])
        #         print('training accuracy : ', hist.history['acc'])
        #         print('\n')
        #         print('## validation loss and acc ##')
        #         print('validation loss : ', hist.history['val_loss'])
        #         print('validation accuracy : ', hist.history['val_acc'])
        #         print('\n')

        best_val_acc = np.max(hist.history['val_acc'])
        index_best_val_acc = hist.history['val_acc'].index(best_val_acc)
        best_val_loss = hist.history['val_loss'][index_best_val_acc]

        print('\n')
        print('## Best result ##')
        print('Final accuracy : ', best_val_acc)
        print('Final loss : ', best_val_loss)

        print('\n')
        plt.subplot(1, 2, 1)
        plt.title("loss")
        plt.plot(hist.history['loss'], 'b-', label="training")
        plt.plot(hist.history['val_loss'], 'r:', label="validation")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title("accuracy")
        plt.plot(hist.history['acc'], 'b-', label="training")
        plt.plot(hist.history['val_acc'], 'r:', label="validation")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print('\n')

        reload_model = load_model(model_path)
        re_loss, re_acc = reload_model.evaluate(X_test, y_test)

        print('\n')
        print('## Model confirm ##')
        print('Accuracy confirm : ', re_acc)
        print('Loss confirm : ', re_loss)

        #         predict_result = reload_model.predict(X_test)
        #         print(predict_result)
        predict_label = reload_model.predict_classes(X_test)
        #         print(predict_label)
        true_label = np.argmax(y_test, axis=1)
        #         print(Y_true)

        confusion_mtx = confusion_matrix(true_label, predict_label)
        print('\n')
        print(confusion_mtx)
        plt.matshow(confusion_mtx)
        plt.colorbar()
        plt.show()

        true_label_ = pd.Series(true_label, name="Actual")
        predict_label_ = pd.Series(predict_label, name="Predicted")

        df_confusion = pd.crosstab(true_label_, predict_label_)
        #         print(df_confusion)

        kappa_value = cohen_kappa_score(true_label, predict_label)
        #         kappa_value_w = cohen_kappa_score(true_label, predict_label, weights = "quadratic")
        print('\n')
        print('## Kappa value ##')
        print(kappa_value)
        print('\n')
        #         print(kappa_value_w)

        # Result save
        save_file_name = MODEL_SAVE_FOLDER_PATH[2:15] + "-" + file_base + "-" + "fold" + str(fold_no)

        df_confusion.to_csv('./_Classification_result/' + save_file_name + '-confusion_matrix.csv')

        np.savez('./_Classification_result/' + save_file_name + '-variables',
                 training_acc=result_training_acc, training_loss=result_training_loss,
                 validation_acc=result_validation_acc, validation_loss=result_validation_loss,
                 best_acc=best_val_acc, best_loss=best_val_loss,
                 classified_label=predict_label, true_label=true_label,
                 kappa_value=kappa_value)

        fold_no = fold_no + 1

