# Time Series analysis, feature engineering and clustering based on two cases
The repository is aimed to test some specifics of working with TimeSeries data on 2 cases.


The repository covers the following TimeSeries topics on real cases:
1. TimeSeries analysis. tSNE, PCA.
2. Feature engineering: Fast Fourier Transformation, Power Spectral Density, Wavelet transformation, Time Series features autogeneration.
3. Clustering: TimeSeriesKMeans (euclidean and dynamic time warping).

## Cases:
1. Epileptic Seizure Recognition (Classification) based on brain electroencefalography (EEG) results. See details [here](https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition). An epilepsy seizure is characterized by abnormal brain activity, which can be detected using EEG. Dataset containts 11 500 rows with 178 time series points (whcih equal to 1 second og EEG). See notebook `epileptic-seizure-recognition.ipynb`.
2. Cryptocurrency clusterization: find cryptocurrencies whcih differs from bitcoin (__BTC__) in it's behaviour. Data are exported by API. See notebook `crypto-clusterization.ipynb`.

# Author
Parshin Sergei / @ParshinSA / Sergei.A.P@yandex.com
