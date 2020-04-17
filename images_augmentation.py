import imgaug.augmenters as iaa

def stochastic_image_augmentation(image):
    sometimes = lambda aug: iaa.Sometimes(0.4, aug) #apply the augmenter in x% of cases
    more_often = lambda aug: iaa.Sometimes(0.6, aug)
    seq = iaa.Sequential(
        [
            iaa.OneOf([
                iaa.Fliplr(p=0.8), # horizontally flip 60% of all images
                iaa.Flipud(p=0.8), # vertically flip 50% of all images
            ]),
            
            # crop images by -5% to 10% of their height/width
            more_often(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_cval=(0, 255)
            )),
            
            more_often(iaa.OneOf([
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    ])),
            
            sometimes(iaa.OneOf([
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.GammaContrast((0.03, 0.15), per_channel=0.5), # improve or worsen the contrast
            ])),
            
            sometimes(
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)),
            ])),
                        
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                fit_output=True,
            )),
            
        ],
        random_order=True
    )
    #seq_det = seq.to_deterministic() I think this is used if we want to apply the same augmentation of each batch
    return seq.augment_image(image) # return for each image another combinaison 


def do_images_augmentation(X_train, label_train, N):
    
    X_train_aug = [stochastic_image_augmentation(image) for i in range(N) for image in X_train]
    label_train_aug = label_train * N 
    
    assert (len(X_train_aug) == len(label_train_aug)) # making sure the augmentation is done both on images and labels
    
    input_train_aug, label_train_aug = X_train + X_train_aug, label_train + label_train_aug
    
    print('Data augmentation done: initial train size of {} vs output size of {}'.format(len(X_train), len(input_train_aug)))
    return input_train_aug, label_train_aug

def show_random_images(list_images, list_labels, N_to_show):
    # this function is mostly for verification 
    assert(len(list_images) == len(list_labels))
    sequence = range(len(list_images))
    rnd_to_show = random.choices(sequence, k=N_to_show)
    
    for i in rnd_to_show: 
        pyplot.figure()
        print(list_labels[i])
        pyplot.imshow(list_images[i])
        pyplot.show()