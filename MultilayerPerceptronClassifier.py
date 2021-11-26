def MLP_model_5_1_1(ff1_fft_equal_duration_dataset_vectorized):

    labelization_df = ff1_fft_equal_duration_dataset_vectorized
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    sc = spark.sparkContext

    schema_output = T.StructType([
        T.StructField("model_id", T.StringType(), True),
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
    label_column_list =[xx for xx in label_column_list  if "ff1_label_luyn" not in xx]
    label_column_list =[xx for xx in label_column_list  if "ff1_label_baptiste" not in xx]
    label_column_list =[xx for xx in label_column_list  if "ff1_label_tran" not in xx]

    for label_column in label_column_list:
        print("-" + label_column)

        # Filter data from input dataset (same source, same len)
        input_df = labelization_df.filter(~F.isnull(F.col(label_column)))

        print ("Row number = " + str(input_df.count()))


        # Split train vs test datasets
        train_df, test_df = input_df.randomSplit([0.8, 0.2], seed=1)

        # specify layers for the neural network:
        # input layer of size (features)
        # intermediate(s) of size 
        # output of size (classes)
        layers = [120, 10, 20, 2]
        iterations = 150

        # Set model description for log
        model_desc = "MultilayerPerceptronClassifier - Layers " + str(layers) + " - Iterations: " + str(iterations) + " - label: " + str(label_column)


        # create the trainer and set its parameters
        trainer = MultilayerPerceptronClassifier(maxIter=iterations, layers=layers, blockSize=128, seed=1234, featuresCol='ff1_seq', labelCol=label_column)

        # train the model
        model = trainer.fit(train_df)

        # compute accuracy on the train set
        result = model.transform(train_df)
        predictionAndLabels = result.select("prediction", label_column)
        # Display results
        train_bool = True
        record_tmp = result_prediction2(result, label_column, "prediction", train_bool, model_desc)
        record = record + record_tmp

        # compute accuracy on the test set
        result = model.transform(test_df)
        predictionAndLabels = result.select("prediction", label_column)
        # Display results
        train_bool = False
        record_tmp = result_prediction2(result, label_column, "prediction", train_bool, model_desc)
        record = record + record_tmp

    record_df = spark.createDataFrame(record, schema_output)

    return record_df




    """
    # example from pyspark documentation
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Load training data
    data = spark.read.format("libsvm")\
        .load("data/mllib/sample_multiclass_classification_data.txt")

    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4], 1234)
    train = splits[0]
    test = splits[1]

    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [4, 5, 4, 3]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

    # train the model
    model = trainer.fit(train)

    # compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    """

