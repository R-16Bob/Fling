from easydict import EasyDict
import torch

#data_dir='./drive/MyDrive/FL_data/Fling_data'  # Colab的数据保存路径
data_dir='./data/'  # 正常的数据保持路径
exp_args = dict(
    data=dict(  # 对数据集的设置
        dataset='cifar10', data_path='./data/CIFAR10', sample_method=dict(name='dirichlet', train_num=500, test_num=100,alpha=0.1)
    ),  # 这个路径是存储数据集的路径
    learn=dict(  # 对学习的设置
        device='cuda:0' if torch.cuda.is_available() else "cpu", local_eps=8, global_eps=300, batch_size=100, optimizer=dict(name='sgd', lr=0.02, momentum=0.9)
    ),
    model=dict(  # 对模型的设置
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=40),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=1, logging_path=data_dir+'/logging/cifar10_fedavg_resnet_dir_0.1')  # 这是存储的Log的路径
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)
