gpu=$1

cd ..

for arc in {0..923}; do
    python train_student2.py --gpu $gpu --dataset mnist --distill kd --model_s fc5 -r 1 -a 0 -b 0 --trial 1 --epochs 100 --arc $arc --prefix mnist/base --val_freq 10;
done
