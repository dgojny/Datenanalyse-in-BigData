# Use an official Python image as a parent image
FROM openjdk:8

# Set environment variables
ENV SPARK_VERSION=3.5.0 \
    HADOOP_VERSION=3 \
    PYSPARK_PYTHON=python3 \
    PYTHON_VERSION=3.10

# Set environment variables to non-interactive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies, add the deadsnakes PPA, and install Python 3.10
RUN apt-get update && \
    apt-get install -y wget build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    libgdbm-dev libnss3-dev libedit-dev libuuid1 libdb-dev uuid-dev && \
    wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.0 Python-3.10.0.tgz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Install Jupyter
RUN pip3.10 install jupyterlab

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
RUN pip install pyspark findspark numpy scipy pandas

WORKDIR /notebooks

# Expose Jupyter port
EXPOSE 8888

# Define the entrypoint for Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
