# Emotion Recognition

## Datasets
Images containing labels of expressions
1) FER2013: The dataset consistes of a csv file csv file. The first column of the csv represents the labels of expressions. They are int numbers ranging from 0 - 6 where 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. The second column contains the pixel values of the image. The `data_processor.py` will take care of that. It loads the cvs, captures the labels, captures the pixels, reshape the pixels and save them to the corresponding labels directory located `fer2013/processed`.</br>
```
.
|___fer2013
|   |    fer2013.csv
|   |    processed
|   |     |____test
|   |     |     |____0
|   |     |     |____1
|   |     |     |____2
|   |     |____train
|   |            |____0
|   |            |____1
|   |            |____2
|    main.py
|    models.py
|    data_processor.py
```

2) CK+: The dataset consists of ~500 subjects and consists of 7 expressions. Every subject has all the expressions. These expressions are sequences of expressions from netural to the peak of that specific class of expression. It is also an image dataset and labels are available as well as the landmarks of the face.

Note that all the data needs to be in the above mentioned format in the processed directory



