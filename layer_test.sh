#!/bin/bash
set -e

rm -rf /tmp/ml_base_tests

ACTIV=('identity' 'relu')
CONV_NORMALIZATION=('batchnorm' 'groupnorm' 'spectralnorm' 'none')
DENSE_NORMALIZATION=('batchnorm' 'spectralnorm' 'none')

echo "testing resnet layers..."
for i in 32 64 128; do
    CUR_ACTIV=${ACTIV[RANDOM%${#ACTIV[@]}]}
    CUR_DENSE_NORMALIZATION=${DENSE_NORMALIZATION[RANDOM%${#DENSE_NORMALIZATION[@]}]}
    CUR_CONV_NORMALIZATION=${CONV_NORMALIZATION[RANDOM%${#CONV_NORMALIZATION[@]}]}
    python vae_main.py --vae-type=simple --conv-normalization=${CUR_CONV_NORMALIZATION} --activation=${CUR_ACTIV} --dense-normalization=${CUR_DENSE_NORMALIZATION} --lr-update-schedule=cosine --warmup=0 --optimizer=lars_adam --lr=1e-3 --decoder-layer-type=resnet --encoder-layer-type=resnet --reparam-type=isotropic_gaussian --kl-beta=1.0 --task=binarized_mnist --data-dir=$HOME/datasets/binarized_mnist --batch-size=3 --epochs=2 --nll-type=l2 --continuous-size=64 --debug-step --polyak-ema=0 --weight-decay=1e-6 --monte-carlo-posterior-samples=2 --model-dir=/tmp/ml_base_tests --log-dir=/tmp/ml_base_tests --kl-annealing-cycles=3 --image-size-override=$i --max-time-steps=4 --jit --uid=ajtv00_$i || exit 1
done


echo "testing conv layers..."
for i in 28 32 64 128; do
    CUR_ACTIV=${ACTIV[RANDOM%${#ACTIV[@]}]}
    CUR_DENSE_NORMALIZATION=${DENSE_NORMALIZATION[RANDOM%${#DENSE_NORMALIZATION[@]}]}
    CUR_CONV_NORMALIZATION=${CONV_NORMALIZATION[RANDOM%${#CONV_NORMALIZATION[@]}]}
    python vae_main.py --vae-type=simple --conv-normalization=${CUR_CONV_NORMALIZATION} --activation=${CUR_ACTIV} --dense-normalization=${CUR_DENSE_NORMALIZATION} --lr-update-schedule=cosine --warmup=0 --optimizer=lars_adam --lr=1e-3 --decoder-layer-type=conv --encoder-layer-type=conv --reparam-type=isotropic_gaussian --kl-beta=1.0 --task=binarized_mnist --data-dir=$HOME/datasets/binarized_mnist --batch-size=3 --epochs=2 --nll-type=l2 --continuous-size=64 --debug-step --polyak-ema=0 --weight-decay=1e-6 --monte-carlo-posterior-samples=2 --model-dir=/tmp/ml_base_tests --log-dir=/tmp/ml_base_tests --kl-annealing-cycles=3 --image-size-override=$i --max-time-steps=4 --jit --uid=ajtv00_$i || exit 1
done


echo "testing dense layers..."
# only need to test one size since logic is all the same
for i in 32; do
    CUR_ACTIV=${ACTIV[RANDOM%${#ACTIV[@]}]}
    CUR_DENSE_NORMALIZATION=${DENSE_NORMALIZATION[RANDOM%${#DENSE_NORMALIZATION[@]}]}
    CUR_CONV_NORMALIZATION=${CONV_NORMALIZATION[RANDOM%${#CONV_NORMALIZATION[@]}]}
    python vae_main.py --vae-type=simple --conv-normalization=${CUR_CONV_NORMALIZATION} --activation=${CUR_ACTIV} --dense-normalization=${CUR_DENSE_NORMALIZATION} --lr-update-schedule=cosine --warmup=0 --optimizer=lars_adam --lr=1e-3 --decoder-layer-type=dense --encoder-layer-type=dense --reparam-type=isotropic_gaussian --kl-beta=1.0 --task=binarized_mnist --data-dir=$HOME/datasets/binarized_mnist --batch-size=3 --epochs=2 --nll-type=l2 --continuous-size=64 --debug-step --polyak-ema=0 --weight-decay=1e-6 --monte-carlo-posterior-samples=2 --model-dir=/tmp/ml_base_tests --log-dir=/tmp/ml_base_tests --kl-annealing-cycles=3 --image-size-override=$i --max-time-steps=4 --uid=ajtv00_$i || exit 1
done

echo "\ntests complete! :)\n"
