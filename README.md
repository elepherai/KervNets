# KervNets
Kervolutional Neural Networks

# Usage:
## 1. MNIST

    python le_mnist.py
    
## 2. CIFAR-10/100

    python cifar-10.py --log cifar-10-kres34
    

## 3. Imagenet

    python imagenet_train.py IMAGENET_PATH
    
### Resume
    python imagenet_train.py --start-epoch 60 --lr 0.0001 --resume RECORD_PATH IMAGENET_PATH

### Performance on ImageNet

|     Model     |  Top 1 error  |  Top 5 error  | Top 1 10-crop | Top 5 10-crop |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Resnet-18     |     30.24     |     10.92     |     27.88     |      9.42     |
| KResnet-18    |     29.74     |     10.49     |     27.43     |      9.03     |
| Resnet-34     |     26.70     |      8.58     |     25.03     |      7.76     |
| KResnet-34    |     26.29     |      8.34     |     24.28     |      7.08     |
| Resnet-50     |     23.85     |      7.13     |     22.85     |      6.71     |
| KResnet-50    |     23.56     |      6.90     |     22.05     |      5.97     |
| Resnet-101    |     22.63     |      6.44     |     21.75     |      6.05     |
| KResnet-101   |     21.63     |      6.16     |     20.92     |      5.18     |



