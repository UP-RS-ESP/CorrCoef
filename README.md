# CorrCoef
Python C-extension for memory efficient and multithreaded Pearson product-moment correlation coefficient estimation using OpenMP.

## Install

	git clone https://github.com/Rheinwalt/CorrCoef.git
	cd CorrCoef
	sudo python setup.py install

## Example
An example using random data for `num = 4` time series of length `len = 100` using as many threads as possible:

	import numpy as np
	import CorrCoef

	num = 4
	len = 100
	data = np.random.random((num, len))
	corr = CorrCoef.Pearson(data)

