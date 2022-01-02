import os
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, StringIndexer, VectorAssembler, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import sys

class SpamClassifier_pyspark():
  if __name__ == '__main__':
    input_file = sys.argv[-1]
    spark = SparkSession.builder.appName('SpamClassifier').getOrCreate()

    data = spark.read.csv(input_file, header=True, inferSchema=True, sep='\t')
    data = data.na.drop(subset=["text"]) # drop null columns & rows

    tokenizer = Tokenizer(inputCol="text", outputCol="token_text")   # tokenize input string
    stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')  # filter out stop words from input
    count_vector = CountVectorizer(inputCol="stop_tokens", outputCol="token_features")  # convert a collection of text documents to vectors of token count
    indexer = StringIndexer(inputCol="class", outputCol="label")   # converts a single column to an index column
    vec_assembler = VectorAssembler(inputCols=['token_features'], outputCol="features")  # merge vectors into a single feature vector

    data_pip = Pipeline(stages=[tokenizer,stopremove,count_vector,indexer,vec_assembler]) # chains multiple transformers to create a ML workflow
    new_data = data_pip.fit(data).transform(data)

    train, test = new_data.randomSplit([0.7, 0.3], seed = 2026) # split into traning and testing sets
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train)  # naive bayes fitting
    predict = model.transform(test)
    predict_table = predict.select("label", "prediction", "probability")
    eval = BinaryClassificationEvaluator(rawPredictionCol="prediction")  # binary classifier
    acc = eval.evaluate(predict)
    print("Naive Bayes Model Accuracy (Binary Classification): ", acc)

    eval2 = MulticlassClassificationEvaluator()   # multi-class classifier
    acc2 = eval2.evaluate(predict)
    print("Naive Bayes Model Accuracy (Multi-class Classification): ", acc2)
    print("Example predicted output:")
    predict_table.show()