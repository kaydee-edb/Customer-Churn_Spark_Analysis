# Customer Churn Analysis

The repository provides insights on a simple use case of Customer Churn Analysis. At my current work place, I have developed and deployed Customer Churn Analysis based on healthcare subscription data. However, Since i could not use the same dataset, I have picked a customer churn data set at a telecom provider. 

The below is the schema of the dataset as read by spark:

root
 |-- churn: string (nullable = true)
 |-- accountlength: integer (nullable = true)
 |-- internationalplan: string (nullable = true)
 |-- voicemailplan: string (nullable = true)
 |-- numbervmailmessages: integer (nullable = true)
 |-- totaldayminutes: double (nullable = true)
 |-- totaldaycalls: integer (nullable = true)
 |-- totaldaycharge: double (nullable = true)
 |-- totaleveminutes: double (nullable = true)
 |-- totalevecalls: integer (nullable = true)
 |-- totalevecharge: double (nullable = true)
 |-- totalnightminutes: double (nullable = true)
 |-- totalnightcalls: integer (nullable = true)
 |-- totalnightcharge: double (nullable = true)
 |-- totalintlminutes: double (nullable = true)
 |-- totalintlcalls: integer (nullable = true)
 |-- totalintlcharge: double (nullable = true)
 |-- numbercustomerservicecalls: integer (nullable = true)

As we could see, most of the predictor variables are numeric and some are of categorical nature. Spark did not have any issue infering the schema for this dataset as the dataset is provided in a csv format. We often run into situations where spark might read majority of the variables as string, due to the values in the input data. I have covered what we can do in such situation as part of the code Notebook. 

Finally, I stopped short of doing feature engineering and building additional models as I wanted this to be an introduction to Spark ML capabilities. I will update the notebook in future. 

Thanks for reading!

