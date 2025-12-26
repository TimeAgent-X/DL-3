# DL-3

## Datasets

在项目根目录新建data文件夹，将数据集放在data文件夹中。

[Yelp Phoenix Academic Dataset](https://github.com/rekiksab/Yelp-Data-Challenge-2013/tree/master/yelp_challenge/yelp_phoenix_academic_dataset)

## Run Experiments

Option 1: Run All Experiments Automatically

```bash
./run_experiments.sh
```
OR

```bash
# Add --limit_samples N if you want to test on a subset (--limit_samples 5000)
python main.py --mode tune --epochs 10
```




Option 2: Run Experiments Individually
```bash
# 1. Base Configuration
python main.py --mode train --save_path model_Base.pth --plot_path curve_Base.png --epochs 10

# 2. Dropout Analysis
python main.py --mode train --save_path model_Dropout_0.2.pth --plot_path curve_Dropout_0.2.png --epochs 10 --dropout 0.2

python main.py --mode train --save_path model_Dropout_0.8.pth --plot_path curve_Dropout_0.8.png --epochs 10 --dropout 0.8

# 3. Layer Normalization
python main.py --mode train --save_path model_With_LayerNorm.pth --plot_path curve_With_LayerNorm.png --epochs 10 --use_layer_norm

# 4. Learning Rate Decay
python main.py --mode train --save_path model_With_LR_Decay.pth --plot_path curve_With_LR_Decay.png --epochs 10 --lr_decay

# 5. Depth and Residuals
python main.py --mode train --save_path model_Depth_2.pth --plot_path curve_Depth_2.png --epochs 10 --num_layers 2

python main.py --mode train --save_path model_Depth_2_Residual.pth --plot_path curve_Depth_2_Residual.png --epochs 10 --num_layers 2 --use_residual

python main.py --mode train --save_path model_Depth_2_Residual_Norm.pth --plot_path curve_Depth_2_Residual_Norm.png --epochs 10 --num_layers 2 --use_residual --use_layer_norm

```
## Test

```
python test.py
```