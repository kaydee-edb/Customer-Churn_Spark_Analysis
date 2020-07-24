""" 
This routine scores the new test data by using the pipelines saved 
during the training process. This can be exposed as a webservice to 
cater new requests.
"""

# Import required libraries

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from time import ctime
from train import loadData, numImpute, catImpute
from spark import get_spark

#Create SparkSession instance for the run
spark = get_spark()

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

