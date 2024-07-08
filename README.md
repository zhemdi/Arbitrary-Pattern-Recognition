# Arbitrary Pattern Recognition

This repository contains mplementations of ILPONet (Invariant to Local Patterns Orientation Network) and EquiLoPO (Equivariant to Local Patterns Orientation Network) , convolutional neural network architectures that achieve rotational invariance and equivariance for volumetric data analysis respectively. The key components of this repository are:

## Repository Structure

- **ILPONet/**: Contains the ILPONet package, including the implementation of the ILPONet convolution operator and related utilities.

- **EquiLoPO/**: Contains the EquiLoPO package, including the implementation of the EquiLoPONet convolution operator and related utilities.

- **MedMNIST/**: This directory contains adaptations of ResNet-18 and ResNet-50 architectures that utilize the ILPONet convolution operator. The adaptations are designed to work with the MedMNIST dataset, a comprehensive resource for benchmarking machine learning models on medical image datasets. The MedMNIST codebase is sourced from [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST) and is licensed under the Apache License 2.0.

- **se3cnn/**: Incorporates adaptations of ResNet-50 architectures leveraging the ILPONet convolution operator, tailored for compatibility with se3cnn, a library for 3D rotation equivariant CNNs. The se3cnn code utilized here is based on a specific commit from the [se3cnn GitHub Repository](https://github.com/mariogeiger/se3cnn/tree/546bc682887e1cb5e16b484c158c05f03377e4e9), which is under the MIT License.

## Usage

### Example Training Commands

Below are example commands for training ILPONet on the CATH and MedMNIST datasets:

#### For se3cnn:

```bash
python3 /se3cnn/experiments/scripts/cath/cath.py \
  --data-filename /se3cnn/data/cath_10arch_ca.npz \
  --model ILPOResNet34Smallhardmax \
  --training-epochs 100 \
  --batch-size 4 \
  --batchsize-multiplier 1 \
  --randomize-orientation \
  --kernel-size 3 \
  --initial_lr=0.0005 \
  --lr_decay_start=40 \
  --burnin-epochs 40 \
  --lr_decay_base=.94 \
  --downsample-by-pooling \
  --p-drop-conv 0.01 \
  --report-frequency 1 \
  --lamb_conv_weight_L1 1e-7 \
  --lamb_conv_weight_L2 1e-7 \
  --lamb_bn_weight_L1 1e-7 \
  --lamb_bn_weight_L2 1e-7 \
  --report-on-test-set
```
### For MedMNIST:
```bash
python3 /MedMNIST/experiments/MedMNIST3D/train_and_eval_pytorch.py \
  --data_flag adrenalmnist3d \
  --conv Conv3d \
  --model_flag ilporesnet18 \
  --download \
  --output_root /MedMNIST/experiments/MedMNIST3D/output/ \
  --so3_size 4 \
  --pooling_type softmax \
  --dropout 0.01 \
  --learning_rate 0.005 \
  --batch_size 16 \
```

An example of training EquiLoPO-Net on the MedMNIST dataset:
    
#### Local Trainable Activation
```bash
python3 ./MedMNIST/experiments/MedMNIST3D/train_and_eval_pytorch.py \
    --data_flag adrenalmnist3d \
    --conv Conv3d \
    --model_flag elporesnet18 \
    --download \
    --output_root ./MedMNIST/experiments/MedMNIST3D/output/ \
    --dropout 0.01 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --order 2 \
    --downsample_by_pooling \
    --coefficients_type trainable \
```

#### Local Adaptive Activation

```bash
python3 ./MedMNIST/experiments/MedMNIST3D/train_and_eval_pytorch.py \
    --data_flag adrenalmnist3d \
    --conv Conv3d \
    --model_flag elporesnet18 \
    --download \
    --output_root ./MedMNIST/experiments/MedMNIST3D/output/ \
    --dropout 0.01 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --order 2 \
    --downsample_by_pooling \
    --coefficients_type adaptive \
    --gpu_ids -1 \
```

#### Global Trainable Activation

```bash
python3 ./MedMNIST/experiments/MedMNIST3D/train_and_eval_pytorch.py \
    --data_flag adrenalmnist3d \
    --conv Conv3d \
    --model_flag elporesnet18 \
    --download \
    --output_root ./MedMNIST/experiments/MedMNIST3D/output/ \
    --dropout 0.01 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --order 2 \
    --downsample_by_pooling \
    --coefficients_type trainable \
    --gpu_ids -1 \
    --global_activation \
```


# Licensing

- The ILPONet package is [MIT License](LICENSE).
- The EquiLoPO package is [MIT License](LICENSE).
- The MedMNIST adaptations are based on code licensed under the Apache License 2.0.
- The se3cnn adaptations are based on code licensed under the MIT License.

