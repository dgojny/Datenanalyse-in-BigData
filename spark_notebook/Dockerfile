# Use an official Python image as a parent image
FROM openjdk:8

# Set environment variables
ENV SPARK_VERSION=3.5.0 \
    HADOOP_VERSION=3 \
    PYSPARK_PYTHON=python3 \
    PYTHON_VERSION=3.9

# Install python and other dependencies
RUN apt-get update && \
    apt-get install -y wget python3 python3-pip python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Install Jupyter
RUN pip install jupyterlab

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -O /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /usr/local/ && \
    cd /usr/local && \
    ln -s spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark && \
    rm /tmp/spark.tgz

# Set environment variables for Spark
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Install PySpark and findspark
RUN pip install pyspark findspark

WORKDIR /notebooks

# Expose Jupyter port
EXPOSE 8888

# Define the entrypoint for Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
