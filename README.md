<div align="center">

<h3> Explicit Visual Prompting for Low-Level Structure Segmentations
 </h3> 
 <br/>
  <a href='https://arxiv.org/abs/2303.08524'><img src='https://img.shields.io/badge/ArXiv-2303.08524-red' /></a> 
  <br/>
  <br/>
<div>
    <a target='_blank'>Weihuang Liu <sup> 1</sup> </a>&emsp;
    <a href='https://xishen0220.github.io/' target='_blank'>Xi Shen <sup> 2</sup></a>&emsp;
    <a href='https://www.cis.um.edu.mo/~cmpun/' target='_blank'>Chi-Man Pun <sup>*,1</sup></a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</sup></a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> University of Macau &emsp; <sup>2</sup> Tencent AI Lab &emsp; 
</div>
<br>
<i><strong><a href='https://arxiv.org/abs/2303.08524' target='_blank'>CVPR 2023</a></strong></i>
<br>
<br>
</div>

<p align="center">
  <img width="50%" alt="teaser" src="teaser/teaser.png">
</p>

## Environment
This code was implemented with Python 3.6 and PyTorch 1.8.1. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Demo
```bash
python demo.py --input [INPUT_PATH] --model [MODEL_PATH] --prompt [PROMPT_PATH] --resolution [HEIGHT],[WIDTH] --config [CONFIG_PATH]
```
`[INPUT_PATH]`: input image
`[PROMPT_PATH]`: prompt checkpoint
`[MODEL_PATH]`: backbone checkpoint
`[HEIGHT]`: target height
`[WIDTH]`: target width
`[CONFIG_PATH]`: config file

## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained SegFormer backbone.
    3. Training:
    ```bash
    python train.py --config configs/train/segformer/train_segformer_evp_defocus.yaml 
    ```
4. Evaluation:
```bash
python test.py --config configs/test/test_defocus.yaml  --model mit_b4.pth --prompt ./save/_train_segformer_evp_defocus/prompt_epoch_last.pth
```
5. Visualization:
```bash
python demo.py --input defocus.png --model ./mit_b4.pth --prompt ./save/_train_segformer_evp_defocus/prompt_epoch_last.pth --resolution 320,320 --config configs/demo.yaml
```

## Train
```bash
python train.py --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH] --prompt [PROMPT_PATH]
```

## Models

Please find the pre-trained models [here](https://uofmacau-my.sharepoint.com/:f:/g/personal/mc05379_umac_mo/EneRfHlTuPZCjH-VVBZVQpMBFXqqdRdU6l8a31jo3i5GOA?e=zagGg1).


## Dataset

### Camouflaged Object Detection
- **COD10K**: https://github.com/DengPingFan/SINet/
- **CAMO**: https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6
- **CHAMELEON**: https://www.polsl.pl/rau6/datasets/

### Defocus Blur Detection
- **DUT**: http://ice.dlut.edu.cn/ZhaoWenda/BTBCRLNet.html
- **CUHK**: http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/

### Forgery Detection
- **CAISA**: https://github.com/namtpham/casia2groundtruth
- **IMD2020**: http://staff.utia.cas.cz/novozada/db/

### Shadow Detection
- **ISTD**: https://github.com/DeepInsight-PCALab/ST-CGAN
- **SBU**: https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html


## Citation

If you find our work useful in your research, please consider citing:

```
@article{zhang2022sadtalker,
  title={SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation},
  author={Zhang, Wenxuan and Cun, Xiaodong and Wang, Xuan and Zhang, Yong and Shen, Xi and Guo, Yu and Shan, Ying and Wang, Fei},
  journal={arXiv preprint arXiv:2211.12194},
  year={2022}
}
```

## Acknowledgements

EVP code borrows heavily from [LIIF](https://github.com/yinboc/liif), [SETR](https://github.com/fudan-zvg/SETR) and [SegFormer](https://github.com/NVlabs/SegFormer). We thank the author for sharing their wonderful code. 
