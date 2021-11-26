def lr_model(labelization):

    labelization_df = labelization
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import LogisticRegression
    from sklearn.metrics import confusion_matrix


    sc = spark.sparkContext

    schema_output = T.StructType([
        T.StructField("model_id", T.StringType(), True),
        T.StructField("source", T.StringType(), True),
        T.StructField("label_used", T.StringType(), True),
        T.StructField("train_bool", T.BooleanType(), True),
        T.StructField("confusion_matrix", T.StringType(), True),
        T.StructField("false_positive_rate", T.DoubleType(), True),
        T.StructField("recall_false", T.DoubleType(), True),
        T.StructField("recall_true", T.DoubleType(), True),
        T.StructField("accuracy", T.DoubleType(), True),
        T.StructField("weighted_false_positive_rate", T.DoubleType(), True),
        T.StructField("weight_precision", T.DoubleType(), True),
        T.StructField("weighted_recall", T.DoubleType(), True)
        ])

    record = []
    label_column_list = [xx for xx in labelization_df.columns if "ff1_label" in xx]
    data_source_list = ["ePoch", "fft"]

    label_column_list =[xx for xx in label_column_list  if "ff1_label_baptiste" not in xx]


    for data_source in data_source_list:
        print (data_source)
        for label_column in label_column_list:
            print("-" + label_column)

            # Filter data from input dataset (same source, same len)
            input_df = labelization_df.filter((F.col("source") == data_source) & ((F.col("desc")) != data_source))\
                .filter(~F.isnull(F.col(label_column)))

            print ("Row number = " + str(input_df.count()))


            # Split train vs test datasets
            train_df, test_df = input_df.randomSplit([0.8, 0.2], seed=1)


            # Set model description for log
            model_desc = "LogisticRegression - data source = " + str(data_source) + " - label = " + str(label_column) 
            lr = LogisticRegression(featuresCol='ff1_seq', labelCol=label_column, maxIter=50, regParam=0.01)

            # Print out the parameters, documentation, and any default values.
            # print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

            # train the model
            model1 = lr.fit(train_df)

            # Since model1 is a Model (i.e., a transformer produced by an Estimator),
            # we can view the parameters it used during fit().
            # This prints the parameter (name: value) pairs, where names are unique IDs for this
            # LogisticRegression instance.
            # print("Model 1 was fit using parameters: ")
            # print(model1.extractParamMap())

            # compute accuracy on the train set
            result = model1.transform(train_df)
            predictionAndLabels = result.select("prediction", label_column)
            # Display results
            train_bool = True
            record_tmp = result_prediction(result, label_column, "prediction", train_bool, model_desc, data_source)
            record = record + record_tmp

            # compute accuracy on the test set
            result = model1.transform(test_df)
            predictionAndLabels = result.select("prediction", label_column)
            # Display results
            train_bool = False
            record_tmp = result_prediction(result, label_column, "prediction", train_bool, model_desc, data_source)
            record = record + record_tmp

    record_df = spark.createDataFrame(record, schema_output)

    return record_df


