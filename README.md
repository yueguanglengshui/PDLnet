# PDLnet
A progressive deep learning framework for fine-grained primate behavior recognition :PDLnet  code

Download the pre-training weights of ResNet-50 from https://www.kaggle.com/datasets/luohongxin/resnet50-pre-training-weight, and place them in "PDLnet/checkpoint".

Download dataset from https://www.kaggle.com/datasets/luohongxin/pbrd-dataset


train_PMG(nb_epoch=20,
         batch_size=12,
         numclass=13,
          store_name='PBRDC',  #粗粒度
         #  store_name='PBRDF',    #细粒度
          train_path='G://datasets//datasets//PBRD//PrimateCoarseGrainedBehavior//train',
          test_path='G://datasets//datasets//PBRD//PrimateCoarseGrainedBehavior//test',
          # train_path='G://datasets//datasets//PBRD//PrimateFineGrainedBehavior//train',
          # test_path='G://datasets//datasets//PBRD//PrimateFineGrainedBehavior//test',
         model_path=None)
         
Modify the parameters in train.py according to your requirements, then run train.py.





