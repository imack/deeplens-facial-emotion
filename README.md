# deeplens-facial-emotion
An AWS Deeplens Challenge Submission Trying to Measure a Viewers Reaction to Funny Media

## The Goal

The main goal of the project is to replace the "focus group" the movie studios would have to perform when a new comedy movie comes out. Instead of having a small group of people from a mall in Iowa determining if something is subjectively funny, our goal is to use the Deeplens to look a a viewers reaction and simply measure their smiles and laughs to more quantitatively determine their reaction to a movie.

To achieve this, we need a project that will identify users in view as they view a movie screener, and a model which will detect their reaction.

Since there is no pre-made model of facial expressions we need to train out own model using AWS Services.

## Training Set Creation

Before we can train a model we need a dataset. Using the [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset, we have 13,000 faces in varios states of expression.

However, we need these labelled to have a "smile" or not and since we didn't really have time to manually check these we decided to make use of [Amazon Rekognition](https://aws.amazon.com/rekognition/) to label the images for us.

While this means our model's accuracy will have an upper bound of Rekognition's accuracy, we have great faith in the Amazon AI team's work.

Using Rekognition, we get a payload of each of the 13,000 images which look like.

```
{
  "Smile": {
    "Value": false,
    "Confidence": 93.7759780883789
  },
  "Emotions": [
    {
      "Type": "HAPPY",
      "Confidence": 91.7003402709961
    },
    {
      "Type": "CALM",
      "Confidence": 12.208562850952148
    },
    {
      "Type": "SAD",
      "Confidence": 9.688287734985352
    }
  ]
}
```

While eventually we'd like to make use of other emotions for things like comedies and what-not, for now we're just focused on smiling and laughing for comedies. However, the Deeplens networks were not about to handle more than one output up until less than a week before due date :-P.

In order to focus the learning of the model on faces, we make us of [OpenCV's Haar Cascades](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html) to crop the images to just the faces, as seen below:

![Al_gore_before](images/Al_Gore_0001.jpg)

![Al_gore_before](images/Al_Gore_resized.jpg)

We wrap up the 13,000 cropped images (split into training and validation sets) into [rec files](https://mxnet.incubator.apache.org/tutorials/basic/image_io.html) for easier training with Gluon. With all the labels we've parsed. Though, we'll just train on smiles for now.

```
python im2rec.py   ~/deeplens-facial-emotion/listfile-train.lst ~/deeplens-facial-emotion/cnn-face/
python im2rec.py   ~/deeplens-facial-emotion/listfile-val.lst ~/deeplens-facial-emotion/cnn-face/
```

All this work can be seen in the `dataset-creation.ipynb` notebook.

## The Custom Model (Smile Net)

With our 2 Rec files for training in validation, we can then train a modified [AlexNet](https://en.wikipedia.org/wiki/AlexNet) to make a Convolutional Neural Network (with the last 2 dense layers lightened to bring down model size) for training.

After 30 epochs and about an 80% accuracy rate, we can then output the model to S3 for use in our Deeplens project.

All this work can be seen in the `face-detection-model-creation` notebook.

## The Lambda Function




## Demo Video



## Further Work

* Obviously a better initial dataset would help training.
* More effort into the emotion model would be useful as it's 80% accuracy probably isn't enough for what we'd want out system to do
* Uploading a runtime of emotional scores to a kinesis streams would be useful in aggregating responses to comedic media.




