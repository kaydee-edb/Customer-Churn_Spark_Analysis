""" 
This is the main routine to train a Logistic Regression Model.The 
final model will be saved as a Pipeline model and utilized to run 
predictions for the new data. 
"""
# Import required libraries

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (OneHotEncoderEstimator, StringIndexer,
                                Imputer, VectorAssembler, 
                                SQLTransformer, QuantileDiscretizer)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import (CrossValidator, CrossValidatorModel, 
                               ParamGridBuilder)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from spark import get_spark 

#Create SparkSession instance for the run
spark = get_spark()

#Create separate lists of all the variables which can be used in the 
#model. Also, best to separate numeric and categorical variables
label = "churn"
all_cols = ['churn', 'accountlength', 'internationalplan', 
            'voicemailplan', 'numbervmailmessages', 'totaldayminutes', 
            'totaldaycalls', 'totaldaycharge', 'totaleveminutes', 
            'totalevecalls', 'totalevecharge', 'totalnightminutes', 
            'totalnightcalls', 'totalnightcharge', 'totalintlminutes', 
            'totalintlcalls', 'totalintlcharge', 
            'numbercustomerservicecalls']
cat_cols = ['internationalplan', 'voicemailplan']
num_cols = list(set(all_cols) - set(cat_cols) - set([label]))

""" 
The below are the function definitions which will be called from the 
main logic below. These functions will allow us to reuse them as part
of the scoring script.
"""

# load Data Function 
def loadData(path):

    df = spark.read.csv(path, sep = ",",
                        inferSchema = True,header = True,
                        nanValue= ' ', nullValue= ' ')
    
    #call the numeric impute funtion
    df = numImpute(df, num_cols)
    
    #call the categorical impute function
    df = catImpute(df, cat_cols)

    #quick check for missing values still in any categorical columns 
    for col in all_cols:
        if col in num_cols or col == label:
            assert df.filter(df[col].isNull()).count()== 0 , \
                "Column {} contains null values".format(col)
        else:
            assert df.filter(df[col] == '').count()== 0 , \
                "Column {} contains null values".format(col)
        
    return df

#Function to impute any numeric variables with mean
def numImpute(inp_df, numcols):

    for col in numcols:
        inp_df = inp_df.withColumn(col, inp_df[col].cast('double'))
        mean_val = inp_df.select(col).agg({col:'mean'}).collect()[0][0]
        inp_df = inp_df.withColumn(col, F.when(inp_df[col].isNull(), \
                            mean_val).otherwise(inp_df[col]))
    
    return inp_df

#Function to impute any categorical variables using the mostfreq approach
def catImpute(inp_df, catcols):

    for col in catcols:
        
        freq_val = inp_df.select(col).groupby(col).count()\
                            .orderBy('count', ascending= False) \
                            .limit(1).collect()[0][0]
        inp_df = inp_df.withColumn(col, F.when((inp_df[col].isNull() | 
                                       (inp_df[col] == '')), freq_val) \
                                        .otherwise(inp_df[col]))

    return inp_df

#This routine will host the pipeline stages
def pipeline_preprocess(inp_df):

    # since categorical variables need both string indexing and 
    # encoder, we can merge both the operations
    
    stages = [] #this will host the steps for our data transformation

    for col in cat_cols:
        stringIndexer = StringIndexer(inputCol= col, \
                                      outputCol=col + "Index")
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer \
                                                    .getOutputCol()], \
                                    outputCols = [col + "catVec"])
        stages += [stringIndexer, encoder]

    acctlen_bin = QuantileDiscretizer(numBuckets=4, \
                                        inputCol = "accountlength", \
                                        outputCol="acctlen_bin")   
    stages += [acctlen_bin]

    # Create the label_Idx for the Output Column
    label_Idx = StringIndexer(inputCol= "churn", outputCol = "label")
    
    stages += [label_Idx]

    # Create the vector assembler stage to assemble data into labels
    # and features
    numeric_cols = num_cols.copy()
    numeric_cols.remove("accountlength")
    numeric_cols.append("acctlen_bin")
       
    inp_features = [c + "catVec" for c in cat_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=inp_features, \
                                outputCol="features")
    stages += [assembler]

    data_preprocess = Pipeline(stages=stages).fit(inp_df)

    return data_preprocess

def train_routine(file_path):

    # load the dataset from the path
    df = loadData(file_path)

    pre_process = pipeline_preprocess(df)
    df = pre_process.transform(df)

    # Save the preprocess model
    pre_process.write().overwrite().save("./output/preprocess")

    """ 
    Fit a Logistic regression model on the training data
    """
    #Initiate a logistic regression object
    lr = LogisticRegression(maxIter=100)

    #Create a paramgrid for Cross Validation
    paramgrid = (ParamGridBuilder()
                .addGrid(lr.regParam,[0.1, 0.01])
                .build())

    #Train the crossvalidator
    cv = CrossValidator(estimator = Pipeline(stages = [lr]),
            estimatorParamMaps=paramgrid,
            evaluator=BinaryClassificationEvaluator(rawPredictionCol="prediction"), 
            numFolds=5)
    cv_model = cv.fit(df)

    #Save the best model 
    cv_model.bestModel.write().overwrite().save("./output/model")

if __name__ == "__main__":

    # Create a path variable which would hold the Input data location
    path = "./input/train/churn.csv"

    train_routine(path)

    print("Pipeline Model has been Saved")


