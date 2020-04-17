# Face_Recognition
Can't remember who's in your class ? This is a fun recognition face tool using CNNs

This repository will cover :
- performing stochastic data augmentation on your images in your train dataset (this prevent from overfitting in classifying)
- using MTCNN to detect faces and crop them
- preparing your initial dataset: split it properly , and fit it to the classifier
- Training different Convolutional Neural network with transfer learning such as VGG 16, Resnet50, by for eg unfreezing different blocks
- Performing predictions on the test set
- Dealing with unknown faces by setting a threshold in the classifier scores 
- Working with classic classifier such as SVM, MLP along with features extractors on images (SIFT, SURF, HOG)
