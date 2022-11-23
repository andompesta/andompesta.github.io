---
layout: post
mathjax: true
title:  "Efficient and scalable machine learning pipelines"
author: "Sandro Cavallari"
tag: "Deep Learning"
comments_id: 3
---

Jobs related to machine learning usually require managing massive datasets.
A well-established rule of thumb that applies to most machine learning projects is that the larger and cleaner the dataset, the better the performance.
Thus, the problem of preprocessing large amounts of data and efficiently feeding the produced dataset into the training pipeline emerges.
While developing fancy models is a fun task for which limitless resources are available on the web, the ML community needs to cover better the topic of streamlining data preprocessing and ingestion pipelines.
Backed by the fast iteration philosophy, this document aims to find the most efficient training process to minimise the cost of experimentation as more experiment results in better performance: as [Elon Musk says](https://www.youtube.com/embed/E7MQb9Y4FAE?start=330&autoplay=1){:target="_blank"}, “high production rate solve many ills”.


To be as general as possible, this article follow the work done by [Chia, P.J., Attanasio, G., Bianchi, F. et al.](ref:chia) and will focus on finetuning a [CLIP-like model](https://openai.com/blog/clip/) on the [farfetch dataset](https://eval.ai/web/challenges/challenge-page/1721/overview).
This task's choice enables us to preprocess a large number of images as well as text, which are the most common data-type currently used in machine learning.

# Data Preprocessing

It is not a secret that the fuel of machine learning applications is data.
For example, online advertising leverage behavioural data to personalise the displayed products, translation services leverage parallel documents, and research engine use users' feedback to learn better query-document rankings.
However, generating these datasets from the raw events is a challenging task.
It is common knowledge that the larger the company, the messier the data; thus, carefully crafted transformation jobs are needed.
Data preprocessing is the tedious step of collecting and integrating events from raw data sources and producing a well-formatted dataset that a training script can consume.
As such, it is usually divided into two stages:

- **Data engineering** is the process of combining different data sources into a prepared dataset. During this process, we want to ensure that data sources are appropriately integrated and aggregated to the right granularity, possible errors are carefully handled, noise examples are removed, and the resulting dataset conforms to a well-defined schema.

- **Feature engineering** aims at converting all the columns of the generated dataset to the appropriate numerical format needed by the model. As such, this process is sometimes postponed to the data loading and augmentation process.

Nowadays, there is a multitude of libraries that can be used for data preprocessing.
[Apache Spark](https://spark.apache.org/){:target="_blank"} appears to be the best tool for data preprocessing jobs.
It provides a Pandas-like interface; it is distributed so it can scale vertically and horizontally, integrated with the most common cloud environments, and supports batch and streaming data sources.
While there a multiple approaches to speed up a preprocessing job, this article only focuses on comparing the different runtime engines supported by Spark:

  - base engine designed to run on CPU clusters used as the baseline,
  - [photon](https://www.databricks.com/product/photon){:target="_blank"} engigne: engine: a C++ implementation of Spark that leverages vectorization operation to reduce the computation time,
  - [rapids](https://www.nvidia.com/en-us/deep-learning-ai/software/rapids/){:target="_blank"} engine: an emerging GPU-based environment provided by NVIDIA.


To compare the different engines, the text preprocessing task is used for benchmarking.
As above mentioned, the dataset is the farfetch dataset that contains about 400K products, but about 300K are used for this test due to missing meaningful descriptions.
Text preprocessing aims to clean, tokenize, pad and add special tokens to each product description.
For consistency, we adopted the same tokenizer used in the original [CLIP implementation](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py).
However, a pyspark UDF is used to apply the preprocessing to the original dataframe:

```python
def encode(value):
    # Petastorm specific enconding function for numpy array datatype.
    # Read following section for a better explanation.
    memfile = io.BytesIO()
    np.save(memfile, value)
    return bytearray(memfile.getvalue())

def get_preprocess_data_fn(
        vocab_path: str
    ) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:

    tokenizer = SimpleTokenizer(
        vocab_path,
        dtype=int,
    )

    def preprocess_data(
        dataframe_batch_iterator: Iterator[pd.DataFrame]
    ) -> Iterator[pd.DataFrame]:
        """
        UDF function to tokenize the items descriptions. Resulting
        numpy array is encoded according to petestorm configuration.
        """

        for dataframe_batch in dataframe_batch_iterator:
            product_ids = []
            descriptions_ids = []
            for row in dataframe_batch.itertuples():
                product_ids.append(int(row.product_id))

                # tokenize description
                description_ids = tokenizer(row.description)
                # description_ids = description_ids.astype(np.int64)
                # enconde numpy array as byte-array
                descriptions_ids.append(encode(description_ids))

            yield pd.DataFrame({
                "product_id": product_ids,
                "description_ids": descriptions_ids,
            })

    return preprocess_data


def main():
    ...
    preprocess_data_fn = get_preprocess_data_fn(
    path=vocab_path
    )

    # generate text encoding and image preprocessing
    farfetch_description = farfetch_description.mapInPandas(
        preprocess_data_fn,
        StructType([
            StructField("product_id", IntegerType()),
            StructField("description_ids", BinaryType()),
        ])
    )
    ...
```
Cluster configuration and experiment results are reported in Tab. [[1]](tab:cluster_definition) and Fig. [[1]](fig:preprocessing_benchmark).

<div style="text-align:center;" id="tab:cluster_definition">
    <p style="font-size:small;">
        Table 1: Clstuer definitions used for data preprocessing benchmarking.
    </p>
    <table>
        <thead>
          <tr>
            <th></th>
            <th>CPU</th>
            <th>Memory</th>
            <th>GPU</th>
            <th>Num Workers</th>
            <th>Price on Databricks (DBU/h)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CPU</td>
            <td>16</td>
            <td>64 GB</td>
            <td>0</td>
            <td>8</td>
            <td>13.7</td>
          </tr>
          <tr>
            <td>Photon-accelerated</td>
            <td>16</td>
            <td>64 GB</td>
            <td>0</td>
            <td>8</td>
            <td>32.6</td>
          </tr>
          <tr>
            <td>GPU-accelerated</td>
            <td>16</td>
            <td>64 GB</td>
            <td>NVIDIA T4</td>
            <td>8</td>
            <td>14.25</td>
          </tr>
        </tbody>
    </table>
</div>

The experiment result shows that a reduction of 31.5 to 49% in processing time achievable just by adopting an accelerated engine.
Rapids is the most promising option as it provides a 2x speedup at a competitive price w.r.t. the CPU baseline.
However, not that in this test, the dataset size is small compared to the available cluster memory, and special care is usually required when working with GPUs as it is pretty common to get out-of-bound memory errors.


<div style="text-align:center;" id="fig:preprocessing_benchmark">
    <figure>
        <img src="{{site.baseurl}}/assets/img/efficent_data_preprocessing/text_preprocessing_benchmark.png" style="max-width: 90%">
        <figcaption style="font-size:small;">
            Figure 1: Data preprocessing benchmark.
        </figcaption>
    </figure>
</div>


# Data Ingestion

Data ingestion is another crucial component for any successfull ML project.
Too little credit have been given to the develipers of [tensorflow Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and [pyTorch DataLoaders](https://pytorch.org/docs/stable/data.html) as they provides common APIs for loading and manipulate the created datasets.
However, these dataloaders utilities are limited when it comes to process large datasets that does not fix the memory and that needs to operate in a distributed environmnet.
On the one hand, Tensorflow Dataset are extreamly efficent when working with TFRecords, but does not support parquet dataformat and requires a significant amount of boilerplate code to be parsed.
On the other hand, pyTorch Dataloaders leave the scope of loading the dataset into the memory for rapid access indexed access to the user.

Due to these limitations, multiple third party solutions arised.
This section focuses on [Petastorm](https://petastorm.readthedocs.io/en/latest/index.html), a general purpous solution provided by Uber which easily integrate into Databricks.
One of the main advantages of Petastorm with respect to other solutions resides in its support of multi-dimentional tensors.
At the core of Petastorm there is the [Codecs](https://petastorm.readthedocs.io/en/latest/_modules/petastorm/codecs.html?highlight=Codec) concept, an API that specify methods to encode and decode custom datatypes.
For example numpy arrays and images, two dtype not supported by Spark, are encoded by Petastorm into a Spark DataFrames as BinaryType and decoded at training time.
As abovementioned, when a new column containing a non-native datatype is added to the DataFrame, the [encode](https://github.com/uber/petastorm/blob/170b22a18ee1c0346d2b289f096804e34a0c5d25/petastorm/codecs.py#L136) function is applied to every row.

```python
def encode(value)
    memfile = BytesIO()
    np.save(memfile, value)
    return bytearray(memfile.getvalue())
```

Similarly, once the dataset is stored on any distributed file system, the Petastorm [Reader](https://github.com/uber/petastorm/blob/170b22a18ee1c0346d2b289f096804e34a0c5d25/petastorm/reader.py#L344) decodes each row of the dataset while feeding the data into the training pipeline.

```python
def decode(value)
    memfile = BytesIO(value)
    return np.load(memfile)
```


Overall, Petastorm appears as a viable solution for use-cases where:
  - the computational cost of the model is order of magnitude higher than loading the data;
  - each example is composed by few columns containing large vectors.

<div style="text-align:center;" id="fig:datasets">
    <figure>
        <img src="{{site.baseurl}}/assets/img/efficent_data_preprocessing/datasets.png" style="max-width: 90%">
        <figcaption style="font-size:small;">
            Figure 2: Visual representations of the different dataset types. Note that for NLP and CV tasks the inputs are raw data format, but the models are extreamly complex. Instead RecSys systems are usualy bounded by the IO oprations related to large amount of users-items pairs; rather than the model complexity. Figure adapted from <a href="https://medium.com/nvidia-merlin/why-isnt-your-recommender-system-training-faster-on-gpu-and-what-can-you-do-about-it-6cb44a711ad4">Why isn’t your recommender system training faster on GPU? (And what can you do about it?)</a>.
        </figcaption>
    </figure>
</div>

As shown in Fig. [[2]](#fig:datasets), this is a common scenario in DeepLearning where NLP or CV applications requires complex models to learn meaningful data representations, but not for reccomandation systems where tabular datasets are the defact standards.
Thus, a separate experiment for tabular data is reported in the next section.

To evaluate Petastorm dataloaders, the dataset previously prepared is used.
The dataset consists of about 300K image/text pairs.
Images are represented as `3 x 224 x 224` arrays, while text by a list of `77` elements.
Each batch is composed by 64 examples.
The objective is to find the best worker-type and number of worker combination as possible.
Thus, a grid search is reported if Fig [[3]](#fig:petastorm_reading), where thread-based and process-based workers are compared with a settings that uses 5, 10 and 20 workers.


<div style="text-align:center;" id="fig:petastorm_reading">
    <figure>
        <img src="{{site.baseurl}}/assets/img/efficent_data_preprocessing/petastorm-reading-benchmark.png" style="max-width: 90%">
        <figcaption style="font-size:small;">
            Figure 3: Dataset loading benchmarking. Upper figure represent the overall execution time for a single epoch, while bottom figure shows the amount of batch per seconds (BpS) processed.
        </figcaption>
    </figure>
</div>


The results show that a settings with 5 processes is the fastes as it is able to process 74 batches per seconds (BpS), which is a 172 % improvement over the default configuration (threaded with 10 workers).


<!-- TODO: -->
<!-- 1) bridge spark to ml-frameworks: -->
<!-- 2) comapre tfrecords (natively supported by Linkeding library) and petastorm parquet   -->

<!--  villan: fast large dataset ingestion from distributed frameworks -->
<!--  why: with extreamly large datasets you need scalable and fast injestion pipelines -->

<!--  how: spark for distributed data preprocessing. benchmark petastorm w.r.t. tfrecords -->

<!--  preprocess large datasets is key, thus the needs for distributed framework. How to efficently feed the generated dataset into your model for training ? -->



# Refereces

<ol>
    <li id="ref:chia"> Chia, P.J., Attanasio, G., Bianchi, F. et al. Contrastive language and vision learning of general fashion concepts. Sci Rep 12, 18958 (2022). https://doi.org/10.1038/s41598-022-23052-9 </li>
</ol>