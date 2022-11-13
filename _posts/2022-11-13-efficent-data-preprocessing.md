---
layout: post
mathjax: true
title:  "Efficent and scalable training pipeline"
author: "Sandro Cavallari"
tag: "Deep Learning"
---

If do any work related to machine lrearning you need to work with datasets and the large and cleaner the dataset the better your performance.
In the last few years, I have faced the joint problem of preprocess large amount of data and efficently feed the produced dataset input my training pipeline.
While developing fancy model is a fun task for which you can find limitless sresources on the web, in my experience it is more important streamline you data preprocessing and injestion pipelines; a topic that is not well covered by the ML-community.
Backed by the fast iteration philosophy, this document aims at find the most efficent training process so to minimise the cost of experimentation as more experiment results in better performance: as [Elon Musk says](https://www.youtube.com/embed/E7MQb9Y4FAE?start=330&autoplay=1){:target="_blank"} "high production rate solve many ills".


# Data Preprocessing

It is not a secret that the fuel of machine learning applications is data.
For example, online advertisment leverage behavioural data to personalise advertisments, translateion services leverage parallel documents while serve engigne uses users feedback to learn better query-document rankings.
However, generating these datasets from the raw events is no easy task. In my experience, the larger the company the messier the data; thus you need carefull crafted trasformation jobs.
Data preprocessing is the tedious step of collecting and integrating events from different raw data-sources and produce a well formated dataset that can be consumed by your training script. As such it is usually divided in two stages:

- **Data engineering** is the process of combine different data-sources into a prepared dataset. During this process we want to ensure that data-sources are properly integrated and aggregated to the right granularity, possible errors are carefully handled, noise examples are removed and the produced dataset conform to a well defined schema.

- **Feature engineering**  is the action of covnerting all the column of the generated dataset to the appropriate numerical format needed by the model. As such, this process is sometimese postponed to the data loading and augmentation process.


Nowadays there are a multitude of libraries that can be used for data preprocess, [Apache Spark](https://spark.apache.org/) appreas to be the best tool for data preprocessing jobs.
It provides a Pandas-like interface, by nature it is distributed thus it can scale vertically and horizontally, it is integrated in the most common clound environments and it supports batch as well as streaming data sources.
While there a multiple approaches to speedup a preprocessing job, this article only focuses in comparing the different runtime engigne supported by spark:

  - base engigne designed to run on CPU clusters ued as baseline,
  - [photon](https://www.databricks.com/product/photon){:target="_blank"} engigne: a C++ implementation of spark that leverage vectorazation operation to reduce the computation time,
  - [rapids](https://www.nvidia.com/en-us/deep-learning-ai/software/rapids/) engigne: an emerging GPU accelerated environment.

To compare the different engigne, we have used 8 datasources forming a total amout of 33 GB of data to preprocess or 100000000 rows to preprocess.
The cluster definitions used for the experiment is reported in Tab. [[1]](#tab:cluster_definition) while the experiment results are shown in Fig [[1]](#fig:preprocessing_benchmark).

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
            <th>Price (DBU/h)</th>
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

Fig. 1 demonstrate how it is possible to obtain a 27.5 to 58% speedup just by adopting an acceleated engigne. Overall, rapids appears as the most promising option as it provite a 2x speedup at a competitive price w.r.t. the CPU baseline.
However, not that in this test the dataset size is small compared to the available cluster memory and special care is usually required  when working with GPUs as it is quite common to get out of bound memory errors.

<div style="text-align:center;" id="fig:preprocessing_benchmark">
    <figure>
        <img src="{{site.baseurl}}/assets/img/efficent_data_preprocessing/preprocessing_benchmark.png" style="max-width: 90%">
        <figcaption style="font-size:small;">
            Figure 1: Data preprocessing benchmark.
        </figcaption>
    </figure>
</div>



<!--  villan: fast large dataset ingestion from distributed frameworks -->
<!--  why: with extreamly large datasets you need scalable and fast injestion pipelines -->

<!--  how: spark for distributed data preprocessing. benchmark petastorm w.r.t. tfrecords -->

<!--  preprocess large datasets is key, thus the needs for distributed framework. How to efficently feed the generated dataset into your model for training ? -->
