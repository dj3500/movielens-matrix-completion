# MovieLens-1M matrix completion

The [MovieLens-1M dataset](https://grouplens.org/datasets/movielens/1m/) has found much use in experiments for machine learning papers.
The dataset contains approximately 1 million ratings for 3900 movies by 6040 users.
The usual experimental setup is to compute a low-rank completion of this user-movie rating matrix, which gives rise to low-dimensional feature vectors for each user and for each movie.
We can then say that the dot product of a user vector with a movie vector approximates the rating of that movie by that user; we can also approximate the similarity of two movies by their (possibly normalized) dot product.

One particular area of application is submodular maximization.
Having the above vectors, we can define useful monotone submodular objective functions
to build a personalized movie recommendation system.
See references below for examples.

This small script, written in 2017 by Jakub Tarnawski (dj3500), reads the user-movie matrix from the MovieLens-1M dataset and computes its low-rank completion (which yields the user and movie vectors).
The rank is set to 20.

## Usage

Ensure that the dependencies are installed. These are:
* Python version at least 3.6
* `scikit-learn`
* `numpy`
* [`fancyimpute`](https://github.com/iskandr/fancyimpute) (can be installed by running `pip install fancyimpute`)

Then:
* download the [MovieLens-1M dataset](https://grouplens.org/datasets/movielens/1m/)
* extract the `ratings.dat` and `movies.dat` files from the archive
* download the `prepare_movies.py` script to the same directory
* run it (`python prepare_movies.py`)

It should take a few minutes and produce files `U.txt` and `VT.txt` in the same directory. These files hold the $U$ (user) and $V$ (movie) matrices, respectively, in text format. The first line holds the number of rows and columns, respectively, and then the matrix is given as space-separated decimal values. The second matrix $V$ is transposed.
The idea is that the MovieLens sparse user-movie matrix $M$ is approximately equal to $M \approx U \cdot V^\top$ (on the entries that are present in $M$).

## References

[1] is the MovieLens-1M dataset.
[2] introduced a facility-location objective function that subsequent works use.
[3-5] use this script in their experiments.

* [1] F. Maxwell Harper, Joseph A. Konstan. [The MovieLens Datasets: History and Context.](http://dx.doi.org/10.1145/2827872) ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)
* [2] Erik M. Lindgren, Shanshan Wu, Alexandros G. Dimakis. [Leveraging Sparsity for Efficient Submodular Data Summarization.](https://arxiv.org/pdf/1703.02690.pdf) NeurIPS 2016
* [3] Slobodan Mitrović, Ilija Bogunović, Ashkan Norouzi-Fard, Jakub Tarnawski, Volkan Cevher. [Streaming robust submodular maximization: A partitioned thresholding approach.](https://arxiv.org/pdf/1711.02598.pdf) NeurIPS 2017
* [4] Ashkan Norouzi-Fard, Jakub Tarnawski, Slobodan Mitrović, Amir Zandieh, Aida Mousavifar, Ola Svensson. [Beyond 1/2-approximation for submodular maximization on massive data streams.](https://arxiv.org/pdf/1808.01842.pdf) ICML 2018
* [5] Marwa El Halabi, Slobodan Mitrović, Ashkan Norouzi-Fard, Jakab Tardos, Jakub Tarnawski. [Fairness in Streaming Submodular Maximization: Algorithms and Hardness.](https://arxiv.org/pdf/2010.07431) NeurIPS 2020
