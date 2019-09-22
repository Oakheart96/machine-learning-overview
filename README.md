# machine-learning-overview
This is the general code for part of the most common implementations of machine learning.

## Common structure

We will describe the common structure of a machine learning program. We won't specifie here all the functions used, because on each machine learning program (Linear Regression, Random Forest, Neural Networks,...) we will go deeper.

### Dataset preparation and preprocessing

The most common types of dataset are csv data (labels and numbers), images or even text. The easy way to read them are

#### CSV file (a.k.a. table data)

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
another way:

```python
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
```

#### Images
```python
def process_data():   
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    images = []
    for each in os.listdir(data_dir):
        images.append(os.path.join(data_dir,each))
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
    content = tf.read_file(images_queue[0])
```

another way:

```python
def load_all(self):
		X = None ; y = None
		if os.path.isfile(self.datafile):
			with open(self.datafile, 'r') as f:
				data = eval(f.readline())
				assert data == self.data, "The image does not match"

			with open(self.datafile, "r") as f:
				lines = f.readlines()

				X = np.zeros((len(lines) - 1, self.target_shape[0]*self.target_shape[1]))
				y = []

				for i in range (len(lines)-1):
					X[i,:] = np.asarray(eval(lines[i + 1].split(self.splitchar)[1]))
					y.append(lines[i + 1].split(self.splitchar)[0])
				y = np.asarray(y)
			return [X, y]

		else:
			raise ValueError("Could not find the datafile")

```

#### Text

```python
with open('file.rst') as File:
    data = File.read()
```

another text file:

```python
data = open('file.txt', 'r').read() 
```

### Machine Learning Algorithms

The 3 different classifications for machine learning algorithms are:
Unsupervised - Kmeans Clustering, Neural Networks,
Semisupervised - We won't study them here.
Reinforcement - Qlearning, Support Vector,Decision Tree, Linear Regression, Logistic Regression, Bayes, Knearest, Neural Networks,...

### Prevent Overfitting

Overfitting is a problem in machine learning algorithms. Sometimes we create a model for our dataset that fits very well, but it doesn't have to much space of margin. 

For example, if we get a regression problem, we might create a funtion that only fit to our dataset. The moment we test it, we fail. That's why is always necessary to test our models.

![equation](https://latex.codecogs.com/gif.latex?w^*&space;=\arg{\min_&space;w&space;\sum_j&space;\big(t(x_j)-\sum_i&space;w_ih_i(x_j)\big)^2&space;&plus;&space;\lambda&space;\sum^k_{i=1}|w_i|^2})

Another way to prevent overfitting in case we are working with images is the dropout.

## Overall dependencies

* numpy
* os
* csv

## Usage

Type `jupyter notebook` into terminal to see the info of each machine learning algorithm. 

We don't have dataset, so you won't be able to run the code inside the notebooks. It's just an info file for you to do your research and adapt it for your own porpouses.