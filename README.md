# Amazon satellite imagery - multi-label classification

Custom model  
https://colab.research.google.com/drive/1RiLQkLJBDJf2sUFcIGyWct2E5FE5mNxT?usp=sharing

EfficientNet, transfer learning  
https://colab.research.google.com/drive/1ojlZrv92zM3OhVUrz4X29kafzO206Fo-?usp=sharing

Multi-label classification of Amazon satellite images.  
Note: 3-Dimensional plots will only render via the above links but not on github.

## Snippet:

Unsupervised learning - clustering - dimensionality reduction:
Lack of distinction between clusters visualizes the difficulty of this classsification task.

![ul3d](https://user-images.githubusercontent.com/79493809/230720566-eb547597-1093-44f6-98f2-50615994aa57.png)

![ul2d](https://user-images.githubusercontent.com/79493809/230720574-2130a756-279e-479a-a96a-4ec359bf8644.png)


## Potential updates:
- include supervised dim reduction, eg. LDA (linear)
- explore libraries that allow plotting images onto 3d space or display image on mousehover

- compute optimal thresholds per class
- other methods of addressing class imbalance: 
  - can try class mode 'sparse'
  - can try 'sample_weight' in tf api instead of 'class_weight' 
  - can try custom weighted binary_cross entropy loss function, multi-label-classification-with-class-weights-in-keras
- can try macro soft f1 loss, eliminates the need to optimize threshold value after training: https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72
- further explore metrics for multi-label classification 

- explore further options for preprocessing images: haze removal, contrast enhancement, image segmentation
- adjust dataset: inspect outliers/maximally dissimilar images, eg. artifacts

- explore computed parameter optimization: tf tuner, optuna, (bayesian optimization)
- explore ensemble networks, allows selecting the most successful model for each label  
- explore ridge regression, adds reguarlization penalty to loss function, reduces issues of multicollinearity, adjusts output to take advantage of label correlations (does not do variable selection)
- try differing learnings rates for each individual layer in transfer learning
- can try train atmospheric and land usage/cover classes seperately
