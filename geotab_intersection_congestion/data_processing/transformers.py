import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, \
    Params, Param, TypeConverters, HasLabelCol, HasPredictionCol, \
    HasFeaturesCol, HasThreshold
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, \
    JavaMLReadable, JavaMLWritable
from pyspark.sql.types import *

__all__ = [
    'NullThresholdRemover'
]


class NullThresholdRemover(Transformer, HasThreshold, HasInputCols,
                           DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCols=None, threshold=0.3) -> None:
        super().__init__()
        self._setDefault(inputCols=inputCols, threshold=threshold)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, threshold=0.3):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        threshold = self.getThreshold()
        if not threshold or threshold > 1.0 or threshold < 0.0:
            raise ValueError("Invalid threshold value")
        cols = dataset.columns
        datasetRowCount = dataset.count()
        inputCols = list(set(self.getInputCols()).intersection(cols))

        colsNullCount = dataset.select(
            [(F.count(F.when(F.col(c).isNull(), c)) / datasetRowCount).alias(c)
             for c in inputCols]).collect()

        colsNullCount = [row.asDict() for row in colsNullCount][0]
        colsGtTh = list(
            {i for i in colsNullCount if colsNullCount[i] > threshold})
        return dataset.drop(*colsGtTh)