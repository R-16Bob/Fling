from easydict import EasyDict
import torch
import os
# os.chdir('../../')  # 在pycharm执行需要先把路径移到Fling目录

exp_args = dict(
    data=dict(dataset='mnist', data_path='./data/mnist', sample_method=dict(name='iid', train_num=500, test_num=100)),
    learn=dict(
        device='cuda:0' if torch.cuda.is_available() else "cpu", local_eps=1, global_eps=10, batch_size=32, optimizer=dict(name='sgd', lr=0.02, momentum=0.9)
    ),
    model=dict(
        name='cnn',
        input_channel=1,
        class_number=10
    ),
    client=dict(name='base_client', client_num=10),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=1, logging_path='./logging/mnist_fedavg_cnn_iid_demo')

    # other=dict(test_freq=1,resume_path='./logging/mnist_fedavg_cnn_iid_demo/model.ckpt',resume_eps=0, logging_path='./logging/mnist_fedavg_cnn_iid_demo')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline

    generic_model_pipeline(exp_args, seed=0)
