# MyoPS2020_CMRsegmentation

 
This repository contains the code to segment the myocardial pathologies in cardiac MRI images for the MyoPS2020 challenge and published as Accurate Myocardial Pathology Segmentation with Residual U-Net.
![myops](https://user-images.githubusercontent.com/44237325/124978200-247f9c80-e03a-11eb-8fcc-037778689add.JPG)
***

Please install essential dependencies (see requirements.txt)

```javascript I'm A tab
numpy==1.19.2
tf_nightly_gpu==2.7.0.dev20210702
ipython==7.25.0
tensorflow==2.5.0
```
***


```CMRsegmentation.ipynb``` contains the code for segmenting the CMRs into the six classes,  left ventricular (LV) blood pool, right ventricular blood pool, LV normal myocardium, LV myocardial edema, LV myocardial scars by using the U-net convolutional neural network architecture built from residual units trained by augmentation operations. The code also contains the data pre-processing to prepare for modeling.


```pipeline.ipynb``` contains the pipeline to segment CMRs into the six classes,  left ventricular (LV) blood pool, right ventricular blood pool, LV normal myocardium, LV myocardial edema, LV myocardial scars with our trained model. The folder of the sample contains a CMR image from [MyoPS 2020 dataset](https://zmiclab.github.io/projects/myops20/data1.html).


***
### MyoPS 2020 Dataset
[MyoPS 2020](https://zmiclab.github.io/projects/myops20/data1.html) challenge dataset consists of three-sequence CMR images from 45 patients. The dataset directly collected from the clinic without any selection. Training dataset consists of 25 cases having a different number of slices of multi-sequence CMR, i.e., late gadolinium enhancement (LGE), T2-weighted CMR which images the acute injury and ischemic regions, balanced-Steady State Free Precession (bSSFP) CMR, and all ground truth values for every single slice. The ground truth labels include left ventricular (LV) blood pool, right ventricular (RV) blood pool, LV normal myocardium, LV myocardial edema, LV myocardial scars and evaluation of the test data will only focus on myocardial pathology segmentation, i.e., scars and edema. The test dataset consists of 20 cases.

### Augmentation Example
We applied several data augmentation techniques to enhance the model generalization ability on unseen datasets. Data augmentation techniques include image dropping out, degree rotation, horizontally flipping, and elastic transformations. The data augmentation method applied to both original images and ground truth masks. To provide expanded dataset, the training dataset was increased ten times.
![segJPG](https://user-images.githubusercontent.com/44237325/125119574-f6fa2800-e0f9-11eb-835d-6ee938a2f5be.JPG)

***

If you find this code base useful, please cite our paper. Thanks!

```
@inproceedings{elif2020accurate,
  title={Accurate Myocardial Pathology Segmentation with Residual U-Net},
  author={Elif, Altunok and Ilkay, Oksuz},
  booktitle={Myocardial Pathology Segmentation Combining Multi-Sequence CMR Challenge},
  pages={128--137},
  year={2020},
  organization={Springer}
}
```

***
Acknowledgement

Mostly the architecture created from this [repository](https://github.com/danielelic/deep-segmentation) by Liciotti et al. was used. 
