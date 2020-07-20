# Customer Churn Analysis

The repository provides insights on a simple use case of Customer Churn Analysis. At my current work place, I have developed and deployed Customer Churn Analysis based on healthcare subscription data. However, Since i could not use the same dataset, I have picked a customer churn data set at a telecom provider. 

The below is the schema of the dataset as read by spark:

>root                                                          \n
> |-- churn: string (nullable = true)                          \n
> |-- accountlength: integer (nullable = true)                 \n
> |-- internationalplan: string (nullable = true)              \n
> |-- voicemailplan: string (nullable = true)                  \n
> |-- numbervmailmessages: integer (nullable = true)           \n
> |-- totaldayminutes: double (nullable = true)                \n
> |-- totaldaycalls: integer (nullable = true)                 \n
> |-- totaldaycharge: double (nullable = true)                 \n
> |-- totaleveminutes: double (nullable = true)                \n
> |-- totalevecalls: integer (nullable = true)                 \n
> |-- totalevecharge: double (nullable = true)                 \n
> |-- totalnightminutes: double (nullable = true)              \n
> |-- totalnightcalls: integer (nullable = true)               \n
> |-- totalnightcharge: double (nullable = true)               \n
> |-- totalintlminutes: double (nullable = true)               \n
> |-- totalintlcalls: integer (nullable = true)                \n
> |-- totalintlcharge: double (nullable = true)                \n
> |-- numbercustomerservicecalls: integer (nullable = true)    \n

As we could see, most of the predictor variables are numeric and some are of categorical nature. Spark did not have any issue infering the schema for this dataset as the dataset is provided in a csv format. We often run into situations where spark might read majority of the variables as string, due to the values in the input data. I have covered what we can do in such situation as part of the code Notebook. 

Finally, I stopped short of doing feature engineering and building additional models as I wanted this to be an introduction to Spark ML capabilities. I will update the notebook in future. 

Thanks for reading!

