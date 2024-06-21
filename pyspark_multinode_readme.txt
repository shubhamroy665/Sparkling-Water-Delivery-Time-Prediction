Install pyspark.
Install h2o,sparkling water.
Command:pip install h2o-3.44.0.2
Download Sparkling water tar file from website. (Website link -  https://s3.amazonaws.com/h2o-release/sparkling-water/spark-3.4/3.46.0.1-1-3.4/index.html)

Dataset Used:
Here we have used zomato dataset from kaggle to predict the delivery time based on different factors.This dataset has 56000 rows.
Dataset title:Zomato Delivery Operations Analytics Dataset.

Go inside the spark folder . 
Command:cd <spark_folder_path>  #In our case it is at /opt/spark.

After that go into the conf folder you will find spark-env.sh file.
Command: cd <spark_folder_path>/conf

For each worker and master node:
spark-env.sh (typically located in the Spark conf directory):
(Put this things using nano or vi editor.)
export SPARK_MASTER_HOST=<master-node-hostname>
export JAVA_HOME=<java_path>
export SPARK_WORKER_CORES=4

For the master node to add the list of workers:
Go into the workers.sh file in the spark/conf folder.

workers (list of worker nodes, typically located in the Spark conf directory):
(Do in  master node)
worker1(ip address)
worker2(ip address)

On the master node, run:(Being in the spark folder)
./sbin/start-master.sh

On each worker node run:(Being in the spark folder)
./sbin/start-workers.sh spark://<master-node-hostname>:7077                #7077 is the port no.(By default it is 7077)

Once the cluster is created:

1.Create a python file.
Here create a spark session followed by h2o context inside of it.
import some dataset, analyse using pyspark and then fit some model using h2o provided libraries.

Then in the master node run the .py file .

At last close the spark session and h2o context inside it.

Now to close the cluster on each worker node run (First go into the spark directory):

./sbin/stop-workers.sh

For master node run:

./sbin/stop-master.sh


