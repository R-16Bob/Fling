from easydict import EasyDict
import torch
import os
# os.chdir('../../')  # 在pycharm执行需要先把路径移到Fling目录
data_dir='./drive/MyDrive/FL_data/Fling_data'  # Colab的数据保存路径

exp_args = dict(
    data=dict(dataset='mnist', data_path='../fuxian/data/mnist', sample_method=dict(name='iid', train_num=500, test_num=100)),
    learn=dict(
        device='cuda:0' if torch.cuda.is_available() else "cpu", local_eps=8, global_eps=40, batch_size=32, optimizer=dict(name='sgd', lr=0.02, momentum=0.9)
    ),
    model=dict(
        name='cnn',
        input_channel=1,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=20),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=3, logging_path='../fuxian/logging/mnist_fedavg_cnn_iid')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline

    generic_model_pipeline(exp_args, seed=0)
