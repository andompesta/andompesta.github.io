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
This choice is done as it will enable us to preprocess a large amount of images as well as text, which are the most common data-type currently used in machine learning.

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


To compare the different engines, the text pre-processing task is used.
The objective of text preprocessing is to clean, tokenize, pad and add special tokens to each prodict description.
For consistency we adopted the same tokenizer used in the original [CLIP implementation](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py).
However, a pyspark UDF is used to apply the preprocessing to the original dataframe:

```python
def encode(value):
    # encode values according to NdarrayCodec
    # https://petastorm.readthedocs.io/en/latest/_modules/petastorm/codecs.html?highlight=bytearray
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

Fig. 1 demonstrate how obtaining a 31.5 to 49% reduction in processing time just by adopting an accelerated engine. Rapids is the most promising option as it provides a 2x speedup at a competitive price w.r.t. the CPU baseline.
However, not that in this test, the dataset size is small compared to the available cluster memory, and special care is usually required when working with GPUs as it is pretty common to get out-of-bound memory errors.

<div style="text-align:center;" id="fig:preprocessing_benchmark">
    <figure>
        <img src="{{site.baseurl}}/assets/img/efficent_data_preprocessing/text_preprocessing_benchmark.png" style="max-width: 90%">
        <figcaption style="font-size:small;">
            Figure 1: Data preprocessing benchmark.
        </figcaption>
    </figure>
</div>


<!-- TODO: -->
<!-- 1) bridge spark to ml-frameworks: -->
<!-- 2) comapre tfrecords (natively supported by Linkeding library) and petastorm parquet   -->

<!--  villan: fast large dataset ingestion from distributed frameworks -->
<!--  why: with extreamly large datasets you need scalable and fast injestion pipelines -->

<!--  how: spark for distributed data preprocessing. benchmark petastorm w.r.t. tfrecords -->

<!--  preprocess large datasets is key, thus the needs for distributed framework. How to efficently feed the generated dataset into your model for training ? -->

## Refereces

<ol>
    <li id="ref:chia"> Chia, P.J., Attanasio, G., Bianchi, F. et al. Contrastive language and vision learning of general fashion concepts. Sci Rep 12, 18958 (2022). https://doi.org/10.1038/s41598-022-23052-9 </li>
</ol>