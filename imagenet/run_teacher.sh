cd ..

python train_student3.py --distill kd --dataset imagenet --batch_size 1024 --model_s MobileNetV2Trofim --part 1 --prefix imagenet/teacher -r 1 -a 0 -b 0 --epochs 100 --val_freq 10 --num_workers 16
