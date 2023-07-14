# canopy-kmeans-mapreduce

Canopy selection is an algorithm where a cheap distance measure is applied for pre-selection. This is a modified machine learning pipeline (Canopy Selection + Kmeans) in the MapReduce framework. However, to make the canopy selection fit in MR, I made a little change to it. This work should be run on Hadoop.

The original paper can be found at: https://dl.acm.org/doi/pdf/10.1145/347090.347123.
