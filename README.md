# deeplens-facial-emotion
An AWS Deeplens Challenge Submission Trying to Measure a Viewers Reaction to Funny Media

## The Goal




## Training Set Creation

13,000

"Labelled Faces in the wild"

http://vis-www.cs.umass.edu/lfw/

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


```
python im2rec.py   ~/deeplens-facial-emotion/listfile-train.lst ~/deeplens-facial-emotion/cnn-face/
python im2rec.py   ~/deeplens-facial-emotion/listfile-val.lst ~/deeplens-facial-emotion/cnn-face/
```

## The Custom Model (Smile Net)

https://en.wikipedia.org/wiki/AlexNet


## The Lambda Function




## Demo Video



## Further Work






