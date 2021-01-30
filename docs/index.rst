.. MLgauge documentation master file, created by
   sphinx-quickstart on Thu Dec 10 22:18:40 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: logo/logo.png
   :align: center

Benchmark ML methods across datasets
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   examples/index
   api

A simple library to benchmark performance of machine learning methods across different datasets. mlgauge is also a wrapper around `PMLB <https://github.com/EpistasisLab/pmlb>`_ which provides benchmark datasets for machine learning.

mlgauge can help you if

* You are developing a machine learning method or an automl system and want to compare and analyze how it performs against other methods.
* You are learning different machine learning methods and would like to understand how different methods behave under different conditions.

Installation
------------

.. code-block:: bash

   pip install mlgauge

Quickstart
----------

To conduct comparative analysis using mlgauge you would need the following:

1. **Methods**: You can use an sklearn API-compliant machine learning method with ``SklearnMethod`` or implement your own using the ``Method`` abstract class.
2. **Datasets**: You can use benchmark datasets available from the PMLB library or pass in your own datasets as a tuple of inputs and labels. You can also pass in a test set.
3. **Metrics**: Define the set of metrics on which the results will be compared. Pass in a callable or a valid sklearn metric string for each metric. If the analysis is not quantitative, the metrics input can be ignored.

``Method`` classes have a ``train`` and ``test`` method which take input datasets and returns multiple metrics. The ``SklearnMethod`` fits sklearn estimators/pipelines during ``train``, returns metrics and optionally exports the model. Each method to be compared has to be initialized before the experiment, they will then be cloned and fit on different datasets.

For example to use sklearn's linear regression and decision tree:

.. code-block:: python

    linear = SklearnMethod(LinearRegression(), metrics=["r2", "max_error"], export_model=False)
    tree = SklearnMethod(DecisionTreeRegressor(), metrics=["r2", "max_error"], export_model=False)

``Analysis`` class runs the experiment but apply each method across datasets. For example to compare the above 2 methods across a range of regression datasets:

.. code-block:: python

    analysis = Analysis(
        methods=[linear, tree],
        metric_names=["r2 score", "max error"],
        datasets="regression",
        n_datasets=10,
        random_state=0,
    )
    analysis.run()

Thanks to integration with PMLB, a random set of 10 regression datasets is automatically collected from the PMLB datasets collection. The datasets used are automatically populated to the ``analysis.datasets`` attribute on initialization.

Once the run completes the results are stored in a ``analysis.result`` attribute as ``DataArrays`` (labeled arrays) of the `xarray <https://xarray.pydata.org/en/stable/>`_ library. The use of labeled arrays provide convenient slicing and dicing of the results of the analysis across datasets, methods, metrics, and splits.

For simple comparison using a ``DataFrame`` table or a plot, the ``analysis.get_results_as_df()`` and ``analysis.plot_results()`` methods are available, respectively.

Refer the examples and API reference to learn more.
