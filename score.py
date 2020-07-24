""" 
This routine scores the new test data by using the pipelines saved 
during the training process. This can be exposed as a webservice to 
cater new requests.
"""

# Import required libraries

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (OneHotEncoderEstimator, StringIndexer, Imputer, 
                                VectorAssembler, SQLTransformer,
                                QuantileDiscretizer)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import (CrossValidator, CrossValidatorModel, 
                               ParamGridBuilder)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from time import ctime
from train import loadData, numImpute, catImpute

#Create SparkSession instance for the run
spark = SparkSession.builder.appName('Score').getOrCreate()

#Load the input Path
path = "./input/test/test.csv"

# Call the function to load data 
df = loadData(path)

#Load the Preprocess Pipeline model which we saved previously
#and process the test data 
test_preprocess = PipelineModel.load("./output/preprocess")

df = test_preprocess.transform(df)

#Load the Cv model which we saved previously
cvModel = PipelineModel.load("./output/model")
preds_labels = cvModel.transform(df)

#Save the preds_lables df into the path specified in the json format
preds_labels.write.format('json') \
            .save("./output/results/" + ctime())

