# Zero-Shot Learning Posed as a Missing Data Problem

This is the MATLAB code of the method 'MDP' in paper [Zero-Shot Learning Posed as a Missing Data Problem](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w38/Zhao_Zero-Shot_Learning_Posed_ICCV_2017_paper.pdf). 
For Python code, please refer to [zero_shot_learning_baseline](https://github.com/PatrickZH/zero_shot_learning_baseline).

## Download the Data
You can download all data (image features, attribtues and word vectors of AwA and CUB) used in this paper 
from [google drive](https://drive.google.com/open?id=18YYOi5FxiBJ5TYLfOkzO3HGw_w-EveyY). 
Then, put the data and code in the same fold (root path of the project).

## Run the Code
Before running the code, you need to install two toolboxes, namely, Dimensionality Reduction toolbox (drtoolbox.tar.gz) and LeastR toolbox (SLEP_package_4.1.zip). 
First, download the two toolboxes from [google drive](https://drive.google.com/open?id=18YYOi5FxiBJ5TYLfOkzO3HGw_w-EveyY). Then, unzip drtoolbox.tar.gz and SLEP_package_4.1.zip. Finally, add the two toolboxes into your Matlab path with subfolders. 

For more support, you can visit [drtoolbox](https://lvdmaaten.github.io/drtoolbox/) and [LeastR](http://www.yelab.net/software/SLEP/). <br>

Now, you can run ICCVW_EXP_AwA.m and ICCVW_EXP_CUB.m !

## Citation
If you use the code or data, please cite our paper:<br>
```
@inproceedings{zhao2017zero,<br>
   title={Zero-Shot Learning Posed as a Missing Data Problem},<br>
   author={Zhao, Bo and Wu, Botong and Wu, Tianfu and Wang, Yizhou},<br>
   booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},<br>
   pages={2616--2622},<br>
   year={2017}<br>
}
```

## Contact
If you have any questions, feel free to contact me.<br>
Email Address: bozhao  at  pku.edu.cn
