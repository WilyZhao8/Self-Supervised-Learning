echo "start to run......"
CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url 'tcp://localhost:16688' --multiprocessing-distributed --loss_loc 0.4 --loss_clu 8.0 --world-size 1 --rank 0 -a resnet50 --lr 0.03 --crop-size 112 --batch-size 1024 --epoch 200 --save-dir outputs/pre_10/ --loss-t 0.3 /media/user/sdisk/Imagenet2012
wait
CUDA_VISIBLE_DEVICES=0,1 python main_lincls.py --dist-url 'tcp://localhost:16689' --multiprocessing-distributed --world-size 1 --rank 0 -a resnet50 --lr 1.0 --batch-size 256 --prefix module.encoder. --pretrained outputs/pre_10/model_best.pth.tar --save-dir outputs/lin_10/ /media/user/sdisk/Imagenet2012




