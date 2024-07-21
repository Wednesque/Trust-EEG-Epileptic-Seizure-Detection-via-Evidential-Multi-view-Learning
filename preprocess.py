                                         from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def split_dataset(x: dict, y, fold=0, num_fold=5):
    assert 0 <= fold < num_fold
    # train
    x_train = dict()
    for k in x.keys():
        x_train[k] = list()
    y_train = list()
    for f in range(num_fold):
        if f != fold:
            for k in x.keys():
                x_train[k].append(x[k][f::num_fold, :])
            y_train.append(y[f::num_fold])
    for k in x.keys():
        x_train[k] = np.concatenate(x_train[k], axis=0)
    y_train = np.concatenate(y_train, axis=0)
    # valid
    x_valid = dict()
    for k in x.keys():
        x_valid[k] = x[k][fold::num_fold, :]
    y_valid = y[fold::num_fold]
    return x_train, y_train, x_valid, y_valid


def process_mat(path=BASE_DIR / 'dataset/handwritten_6views.mat', train=True):
    data = scipy.io.loadmat(path)
    mode = 'train' if train else 'test'
    num_views = int((len(data) - 5) / 2)
    x = dict()
    for k in range(num_views):
        view = data[f'x{k+1}_{mode}']
        x[k] = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)
    y = data[f'gt_{mode}'].flatten().astype(np.int64)
    if min(y) > 0:
        y -= 1
    pickle.dump([x, y], open(path.parent / str(path.stem + f'_{mode}.pkl'), 'wb'))


# deprecated
def process_eeg_extrated_cnn_feature(path=BASE_DIR / 'dataset/eeg/data/feature/fold_1/data_1_train'):
    x = dict()
    num_views = 3
    key_prefix = ('tr' if str(path).find('train') >= 0 else 'te')

    data = scipy.io.loadmat(str(path))
    for k in range(num_views):
        view = data[f'{key_prefix}_X_{k + 1}']
        x[k] = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)
    y = np.argmax(data[f'{key_prefix}_Y'], axis=1)
    pickle.dump([x, y], open(str(path) + '.pkl', 'wb'))


def process_eeg_domain(path=BASE_DIR / 'dataset/eeg/data/domain_feature/train_data1.mat', fold=0, saving_name='dataname'):
    '''
    X_1: 23个通道时域拼接, 23 * 256 dims
    X_2: 23个通道频域4~30HZ, 23 * 27 dims
    X_3: 时-频域, 14 * 256 dims, 2,4,6,...30Hz
    '''
    data = scipy.io.loadmat(str(path))
    x = dict()
    x[0] = MinMaxScaler((0, 1)).fit_transform(data['X'][0][0]).astype(np.float32)
    x[1] = MinMaxScaler((0, 1)).fit_transform(data['X'][1][0]).astype(np.float32)
    x[2] = MinMaxScaler((0, 1)).fit_transform(data['X'][2][0]).astype(np.float32)
    y = np.argmax(data['Y'], axis=1)
    x_train, y_train, x_valid, y_valid = split_dataset(x, y, fold=fold, num_fold=5)
    pickle.dump([x_train, y_train], open(path.parent / f'{saving_name}_fold{fold}_train.pkl', 'wb'))
    pickle.dump([x_valid, y_valid], open(path.parent / f'{saving_name}_fold{fold}_valid.pkl', 'wb'))
    print('---- ', path.name)
    print('domain data shape:', x[0].shape, x[1].shape, x[2].shape, y.shape)
    print('domain training data shape:', x_train[0].shape, x_train[1].shape, x_train[2].shape, y_train.shape)
    print('domain validating data shape:', x_valid[0].shape, x_valid[1].shape, x_valid[2].shape, y_valid.shape)


if __name__ == '__main__':
    # process_mat(BASE_DIR / 'dataset/handwritten_6views.mat', train=True)
    # process_mat(BASE_DIR / 'dataset/handwritten_6views.mat', train=False)

    # for k in range(1, 2):
    #     for person in range(1, 3):
    #         process_eeg_extrated_cnn_feature(BASE_DIR / f'dataset/eeg/data/feature/fold_{k}/data_{person}_train')
    #         process_eeg_extrated_cnn_feature(BASE_DIR / f'dataset/eeg/data/feature/fold_{k}/data_{person}_predict')

    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data1.mat', saving_name='data1')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data2.mat', saving_name='data2')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data3.mat', saving_name='data3')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data4.mat', saving_name='data4')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data5.mat', saving_name='data5')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data6.mat', saving_name='data6')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data7.mat', saving_name='data7')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data9.mat', saving_name='data9')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data10.mat', saving_name='data10')
