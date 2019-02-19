# CMPE257-ImageClassifier_CNN
Image Classifier using CNN

![Image](https://github.com/bhattmaulik1991/CMPE257-ImageClassifier_CNN/blob/master/dogcat.jpeg)

## Approach
For image classification, the primary task is to understand the images. Convolutional Neural Networks are the most popular with image related tasks. 

Below are the steps to perform classification:
  1. Data pre-processing to normalize below irregularities:
    1. Presence of people and other objects in the image
    2. Some images have multiple animals
    3. Images are shot under different amount of light
    4. Some images are blurry
    5. Animals are not always centered in the image
    6. Varying amount of color in the images
    7. Images have different height and width
    
    <img src="https://github.com/bhattmaulik1991/CMPE257-ImageClassifier_CNN/blob/master/1.png" />
    
  2. Constructing a Convolutional Neural Network from scratch for Image classification
    Convolutional Neural Networks are a form of Feedforward Neural Networks. Given below is a schema of a typical CNN. The first part consists of Convolutional and max-pooling layers which act as the feature extractor. The second part consists of the fully connected layer which performs non-linear transformations of the extracted features and acts as the classifier. In the above diagram, the input is fed to the network of stacked Conv, Pool and Dense layers. The output can be a softmax layer indicating whether there is a cat or something else. You can also have a sigmoid layer to give you a probability of the image being a cat. ​a simple stack of 3 convolution layers with a ReLU Our simple CNN model consists of Batch size of 16 neurons was used to train the model. This was too slow in our computer, so we used IBM Hybrid cloud, 16 cores, 32 GB. After running the first experiment with a simple CNN model for 15 epochs, we achieved an accuracy rate of 62.2% with a loss of 93.3%.
  
    <img src="https://github.com/bhattmaulik1991/CMPE257-ImageClassifier_CNN/blob/master/2.png" />
  
  3. Using Data Augmentation techniques for increasing the dataset
    We tried to do image augmentation in the next steps to improve the model accuracy rates. One main concern is to reduce overfitting. So in order to tackle overfitting, we chose to modulate entropic capacity. The main one is the choice of the number of parameters in your model, i.e. the number of layers and the size of each layer. Another way is the use of weight regularization, such as L1 or L2 regularization, which consists in forcing model weights to taker smaller values.
  
    <img src="https://github.com/bhattmaulik1991/CMPE257-ImageClassifier_CNN/blob/master/3.png" />
    
  4. Transfer Learning and fine tuning using pre-trained VGG 19 Network
    The process of training a convolutional neural network can be very time-consuming and require a lot of data. We can go beyond the previous models in terms of performance and efficiency by using a general-purpose, pre-trained image classifier. This example uses VGG16, a model trained on the ImageNet dataset - which contains millions of images classified in 1000 categories. It uses predefined weights in which pretrained model works best.

    <img src="https://github.com/bhattmaulik1991/CMPE257-ImageClassifier_CNN/blob/master/4.png" />
    
## Dataset

Download the dataset [here](https://www.kaggle.com/c/dogs-vs-cats)

Put train data set in below folder structure.

Folder structure

    data/ 
      train/
        
        dogs/ 
            dog001.jpg
            dog002.jpg
            ...
        
        cats/ 
            cat001.jpg
            cat002.jpg
            ...
    
      validation/
        
        dogs/ 
            dog001.jpg
            dog002.jpg
            ...
        
        cats/
            cat001.jpg
            cat002.jpg
            ...

## Conclusion

In this work, we figured out what is Convolutional Neural Network. We assembled and trained the CNN model to classify photographs of dogs and cats. We have tested that this model works really well with a large number of images. We measured how the accuracy depends on the number of epochs in order to detect potential overfitting problem. We determined that 30 epochs are enough for a successful training of the model.
Our next step would be to extend this model on more data sets and try to apply it to multiple categories of animals. We would like to experiment with the neural network design in order to see how a higher efficiency can be achieved in various problems.

    <img src="https://github.com/bhattmaulik1991/CMPE257-ImageClassifier_CNN/blob/master/5.png" />
