# Video Prediction Benchmarks

**We provide benchmark results of video prediction methods on video datasets. More video prediction methods will be supported in the future. Issues and PRs are welcome!**

<details open>
<summary>Currently supported video prediction methods</summary>

- [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NIPS'2015)
- [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NIPS'2017)
- [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
- [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
- [x] [MAU](https://arxiv.org/abs/1811.07490) (CVPR'2019)
- [x] [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2020)
- [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
- [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
- [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)

</details>

<details open>
<summary>Currently supported MetaFormer models for SimVP</summary>

- [x] [ViT](https://arxiv.org/abs/2010.11929) (ICLR'2021)
- [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021)
- [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NIPS'2021)
- [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021)
- [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022)
- [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022)
- [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022)
- [x] [VAN](https://arxiv.org/abs/2202.09741) (ArXiv'2022)
- [x] [IncepU (SimVP.V1)](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- [x] [gSTA (SimVP.V2)](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
- [x] [HorNet](https://arxiv.org/abs/2207.14284) (NIPS'2022)
- [x] [MogaNet](https://arxiv.org/abs/2211.03295) (ArXiv'2022)

</details>

## Moving MNIST Benchmarks

We provide benchmark results on the popular [Moving MNIST](http://arxiv.org/abs/1502.04681) dataset using $10\rightarrow 10$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the final models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Onecycle scheduler.

### **Benchmark of Video Prediction Methods**

For a fair comparison of different methods, we report final results when models are trained to convergence. We provide config files in [configs/mmnist](https://github.com/chengtan9907/SimVPv2/configs/mmnist).

| Method       | Params |  FLOPs | FPS |  MSE  |   MAE  |  SSIM |   Download   |
|--------------|:------:|:------:|:---:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM-S   |  15.0M |  56.8G | 113 | 46.26 | 142.18 | 0.878 | model \| log |
| ConvLSTM-L   |  33.8M | 127.0G |  50 | 29.88 |  95.05 | 0.925 | model \| log |
| PhyDNet      |  3.1M  |  15.3G | 182 | 35.68 |  96.70 | 0.917 | model \| log |
| PredRNN      |  23.8M | 116.0G |  54 | 25.04 |  76.26 | 0.944 | model \| log |
| PredRNN++    |  38.6M | 171.7G |  38 | 22.45 |  69.70 | 0.950 | model \| log |
| MIM          |  38.0M | 179.2G |  37 | 23.66 |  74.37 | 0.946 | model \| log |
| E3D-LSTM     |  51.0M | 298.9G |  18 | 36.19 |  78.64 | 0.932 | model \| log |
| CrevNet      |  5.0M  | 270.7G |  10 | 30.15 |  86.28 | 0.935 | model \| log |
| PredRNN.V2   |  23.9M | 116.6G |  52 | 27.73 |  82.17 | 0.937 | model \| log |
| SimVP+IncepU |  58.0M |  19.4G | 209 | 26.69 |  77.19 | 0.940 | model \| log |
| SimVP+gSTA-S |  46.8M |  16.5G | 282 | 15.05 |  49.80 | 0.967 | model \| log |

### **Benchmark of MetaFormers on SimVP**

Since the hidden Translator in [SimVP](https://arxiv.org/abs/2211.12509) can be replaced by any [Metaformer](https://arxiv.org/abs/2111.11418) block which achieves `token mixing` and `channel mixing`, we benchmark popular Metaformer architectures on SimVP with training times of 200-epoch and 2000-epoch. We provide config file in [configs/mmnist/simvp](https://github.com/chengtan9907/SimVPv2/configs/mmnist/simvp/).

| MetaFormer       |   Setting  | Params |  FLOPs |  FPS |  MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
|------------------|:----------:|:------:|:------:|:----:|:-----:|:-----:|:------:|:-----:|:------------:|
| IncepU (SimVPv1) |  200 epoch |  58.0M |  19.4G | 209s | 32.15 | 89.05 | 0.9268 | 37.97 | model \| log |
| gSTA (SimVPv2)   |  200 epoch |  46.8M |  16.5G | 282s | 26.69 | 77.19 | 0.9402 |  38.3 | model \| log |
| ViT              |  200 epoch |  46.1M | 16.9.G | 290s | 35.15 | 95.87 | 0.9139 | 37.79 | model \| log |
| Swin Transformer |  200 epoch |  46.1M |  16.4G | 294s | 29.70 | 84.05 | 0.9331 | 38.14 | model \| log |
| Uniformer        |  200 epoch |  44.8M |  16.5G | 296s | 30.38 | 85.87 | 0.9308 | 38.11 | model \| log |
| MLP-Mixer        |  200 epoch |  38.2M |  14.7G | 334s | 29.52 | 83.36 | 0.9338 | 38.19 | model \| log |
| ConvMixer        |  200 epoch |  3.9M  |  5.5G  | 658s | 32.09 | 88.93 | 0.9259 | 37.97 | model \| log |
| Poolformer       |  200 epoch |  37.1M |  14.1G | 341s | 31.79 | 88.48 | 0.9271 | 38.06 | model \| log |
| ConvNeXt         |  200 epoch |  37.3M |  14.1G | 344s | 26.94 | 77.23 | 0.9397 | 38.34 | model \| log |
| VAN              |  200 epoch |  44.5M |  16.0G | 288s | 26.10 | 76.11 | 0.9417 | 38.39 | model \| log |
| HorNet           |  200 epoch |  45.7M |  16.3G | 287s | 29.64 | 83.26 | 0.9331 | 38.16 | model \| log |
| MogaNet          |  200 epoch |  46.8M |  16.5G | 255s | 25.57 | 75.19 | 0.9429 | 38.41 | model \| log |
| IncepU (SimVPv1) | 2000 epoch |  58.0M |  19.4G | 209s | 21.15 | 64.15 | 0.9536 | 38.81 | model \| log |
| gSTA (SimVPv2)   | 2000 epoch |  46.8M |  16.5G | 282s | 15.05 | 49.80 | 0.9670 |   -   | model \| log |
| ViT              | 2000 epoch |  46.1M | 16.9.G | 290s | 19.74 | 61.65 | 0.9539 | 38.96 | model \| log |
| Swin Transformer | 2000 epoch |  46.1M |  16.4G | 294s | 19.11 | 59.84 | 0.9584 | 39.03 | model \| log |
| Uniformer        | 2000 epoch |  44.8M |  16.5G | 296s | 18.01 | 57.52 | 0.9609 | 39.11 | model \| log |
| MLP-Mixer        | 2000 epoch |  38.2M |  14.7G | 334s | 18.85 | 59.86 | 0.9589 | 38.98 | model \| log |
| ConvMixer        | 2000 epoch |  3.9M  |  5.5G  | 658s | 22.30 | 67.37 | 0.9507 | 38.67 | model \| log |
| Poolformer       | 2000 epoch |  37.1M |  14.1G | 341s | 20.96 | 64.31 | 0.9539 | 38.86 | model \| log |
| ConvNeXt         | 2000 epoch |  37.3M |  14.1G | 344s | 17.58 | 55.76 | 0.9617 | 39.19 | model \| log |
| VAN              | 2000 epoch |  44.5M |  16.0G | 288s | 16.21 | 53.57 | 0.9646 | 39.26 | model \| log |
| HorNet           | 2000 epoch |  45.7M |  16.3G | 287s | 17.40 | 55.70 | 0.9624 | 39.19 | model \| log |
| MogaNet          | 2000 epoch |  46.8M |  16.5G | 255s | 15.67 | 51.84 | 0.9661 | 39.35 | model \| log |


## TaxiBJ Benchmarks

We provide traffic benchmark results on the popular [TaxiBJ](https://arxiv.org/abs/1610.00081) dataset using $4\rightarrow 4$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the final models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Cosine Annealing scheduler (5 epochs warmup and min lr is 1e-6).

### **Benchmark of MetaFormers on SimVP**

Similar to [Moving MNIST Benchmarks](#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) with training times of 50-epoch. We provide config files in [configs/taxibj/simvp](https://github.com/chengtan9907/SimVPv2/configs/taxibj/simvp/).

| MetaFormer       |  Setting | Params | FLOPs |  FPS |   MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:-----:|:------:|:-----:|:------------:|
| IncepU (SimVPv1) | 50 epoch | 13.79M | 3.61G |  533 | 0.3282 | 15.45 | 0.9835 | 39.72 | model \| log |
| gSTA (SimVPv2)   | 50 epoch |  9.96M | 2.62G | 1217 | 0.3246 | 15.03 | 0.9844 | 39.95 | model \| log |
| ViT              | 50 epoch |  9.66M | 2.80G | 1301 | 0.3171 | 15.15 | 0.9841 | 39.89 | model \| log |
| Swin Transformer | 50 epoch |  9.66M | 2.56G | 1506 | 0.3128 | 15.07 | 0.9847 | 39.89 | model \| log |
| Uniformer        | 50 epoch |  9.52M | 2.71G | 1333 | 0.3268 | 15.16 | 0.9844 | 39.89 | model \| log |
| MLP-Mixer        | 50 epoch |  8.24M | 2.18G | 1974 | 0.3206 | 15.37 | 0.9841 | 39.78 | model \| log |
| ConvMixer        | 50 epoch |  0.84M | 0.23G | 4793 | 0.3634 | 15.63 | 0.9831 | 39.69 | model \| log |
| Poolformer       | 50 epoch |  7.75M | 2.06G | 1827 | 0.3273 | 15.39 | 0.9840 | 39.75 | model \| log |
| ConvNeXt         | 50 epoch |  7.84M | 2.08G | 1918 | 0.3106 | 14.90 | 0.9845 | 39.99 | model \| log |
| VAN              | 50 epoch |  9.48M | 2.49G | 1273 | 0.3125 | 14.96 | 0.9848 | 39.95 | model \| log |
| HorNet           | 50 epoch |  9.68M | 2.54G | 1350 | 0.3186 | 15.01 | 0.9843 | 39.91 | model \| log |
| MogaNet          | 50 epoch |  9.96M | 2.61G | 1005 | 0.3114 | 15.06 | 0.9847 | 39.92 | model \| log |


## WeatherBench Benchmarks

We provide temperature prediction benchmark results on the popular [WeatherBench](https://arxiv.org/abs/2002.00469) dataset (temperature prediction `t2m`) using $12\rightarrow 12$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the final models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Cosine Annealing scheduler (no warmup and min lr is 1e-6).

### **MetaFormers on SimVP for Temperature**

Similar to [Moving MNIST Benchmarks](#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) with training times of 50-epoch. We provide config files in [configs/weather/simvp](https://github.com/chengtan9907/SimVPv2/configs/weather/simvp/).

| MetaFormer       |  Setting | Params | FLOPs |  FPS |  MSE  |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-----:|:------:|:-----:|:------------:|
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 1.238 | 0.7037 | 1.113 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 1.105 | 0.6567 | 1.051 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 1.146 | 0.6712 | 1.070 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 1.143 | 0.6735 | 1.069 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 1.204 | 0.6885 | 1.097 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 1.255 | 0.7011 | 1.119 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 1.267 | 0.7073 | 1.126 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 1.156 | 0.6715 | 1.075 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 1.277 | 0.7220 | 1.130 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 1.150 | 0.6803 | 1.072 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 1.201 | 0.6906 | 1.096 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 1.152 | 0.6665 | 1.073 | model \| log |
