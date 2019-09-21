# machine-learning-overview
This is the general code for part of the most common implementations of machine learning.

## Common structure

We will describe the common structure of a machine learning program. We won't specifie here all the functions used, because on each machine learning program (Linear Regression, Random Forest, Neural Networks,...) we will go deeper.

### Dataset preparation and preprocessing

The most common types of dataset are csv data (labels and numbers), images or even text. The easy way to read them are

```python
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) 
		for row in csvFileReader: file
			dates.append(int(row[row_number].split('-')[0]))
			prices.append(float(row[row_number]))
	return		

```


## Overall dependencies

* numpy

## Usage

Type `jupyter notebook` into terminal and it will run. 

We don't have dataset, so you won't be able to run the code inside the notebooks. It's just an info file for you to do your research and adapt it for your own porpouses.