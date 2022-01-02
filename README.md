
# Using pip or easy_install to install mrjob:
```
pip install mrjob
easy_install mrjob
```

# Using pip to install PySpark:
```
apt-get install openjdk-8-jdk-headless -qq > /dev/null
wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz
tar xf spark-3.0.0-bin-hadoop3.2.tgz
pip install pyspark
pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"

import findspark
findspark.init()
from pyspark.sql import SparkSession
```

# Run 
### Spam Classifier using mrjob
```
python spamClassifier_train.py spam_train_large.csv > spam_train.txt
python spamClassifier_test.py spam_test_large.csv
```

### Spam Classifier using PySpark
```
python SpamClassifier_pyspark.py spam_large.csv
```

## Note
To use the full potential of MRJob and Pyspark by changing the number of nodes/clusters, we recommend that you run the programs on google cloud.
The instruction for installation can be found in the following link: 

https://mrjob.readthedocs.io/en/latest/guides/quickstart.html#installation
