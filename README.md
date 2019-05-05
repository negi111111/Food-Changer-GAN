# Magical Rice Bowl: A Real-time Food Category Changer <br>(ACM MM2018)

The proposed system based on a [StarGAN](https://github.com/yunjey/StarGAN) enables us to change the appearance of a given food photo according to the given category keeping the shape of the given food but exchanging its textures. For training, we used two hundred and thirty thousand food images which achieved very natural food category change among ten kinds of typical Japanese foods: ramen noodle, curry rice, fried rice, beef rice bowl, chilled noodle, spaghetti with meat source, white rice, eel bowl, and fried noodle.

## Ms. Koizumi Loves Ramen Noodles（Japanese Famous Anime）

Ms. Koizumi Loves Ramen Noodles (ラーメン大好き小泉さん Rāmen Daisuki Koizumi-san) is a Japanese manga series by Naru Narumi. It began serialization in Takeshobo's Manga Life Storia magazine in September 2013. A live-action drama series adaptation aired from June 2015 to December 2016. A 12-episode anime television series adaptation co-animated by Studio Gokumi and AXsiZ aired in Japan between January 4 and March 22, 2018. (by [wikipedia](https://en.wikipedia.org/wiki/Ms._Koizumi_Loves_Ramen_Noodles))

<p align="center"><img width="100%" src="./examples/anime.png" /></p>

## Paper

[Magical Rice Bowl: A Real-time Food Category Changer](https://negi111111.github.io/FoodTransferProjectHP/) <br/>
[Ryosuke Tanno](https://github.com/negi111111), [Daichi Horita](https://github.com/UdonDa), [Takumi Ege](https://github.com/ege-t), and [Keiji Yanai](http://acc.cs.uec.ac.jp/yanai/index.html)   <br/>
Department of Informatics, The University of Electro-Communication
<br/>
ACM International Conference Multimedia ([ACM MM](http://www.acmmm.org/2018/)), 2018 (<b>Demo</b>)

<br/>

## Dependencies

- [Python 3.5+](https://www.continuum.io/downloads)
- [PyTorch 0.3.0+](http://pytorch.org/)

<br/>

## Usage

### Pretrained model

This Repository contains the pretrained model in the `./outputs/models` directory.

To translate images using the pretrained model, run the [Google Colaboratory](https://colab.research.google.com/drive/1pZyUwDmQcxWMzqEtFTZ27TxxOw44fsa_). The translated images will be saved into `./outputs/results` directory.

<br/>

## Results
Note that additional results can be see at [FoodTransformation Project HP](https://negi111111.github.io/FoodTransferProjectHP/) and [HoloLens Version](https://negi111111.github.io/FoodChangeLensProjectHP/)

<br/>

## Art Works
- Ryosuke Tanno: Japanese Anime Food Image Transformation, Proc. of NeurIPS Workshop on Machine Learning for Creativity and Design, art works, Motreal, Canada (2018/12). 
  - Please visit AI Art Gallery → [here](http://www.aiartonline.com/design/ryosuke-tanno/) !!.
- Ryosuke Tanno: Food Image Transformation, Proc. of ECCV Workshop on Computer Vision for Fashion, Art and Design, art works, Munich, Germany (2018/9). (longlist) .
  - Please visit Computer Vision Art Gallery Site → [here](https://computervisionart.com/) !!. 
  - My first visual art with computer vision techniques is [here](https://computervisionart.com/pieces/food-image-transformation/) !!. 

<br/>

## Citation

If this work is useful for your research, please cite our [paper](https://negi111111.github.io/FoodTransferProjectHP/):

```
@InProceedings{tann18,
  author={Tanno, R. and Horita, D. and Shimoda, W. and Yanai, K.},
  title={Magical Rice Bowl: Real-time Food Category Changer},
  booktitle={Proc. of ACM International Conference Multimedia},
  year="2018"
}
```

<br/>

## Thanks

This repository is largely based on [StarGAN](https://github.com/yunjey/StarGAN). So, please also cite their [paper](https://arxiv.org/abs/1711.09020).

```
@InProceedings{StarGAN2018,
author = {Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul},
title = {StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
