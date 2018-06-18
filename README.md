# Challenge

Design and implement an MVP that will:
 - Tag each image found in [this](./images.txt) file using the HTTP API (docs), and store the results in an in-memory data structure. Note: Do not use the search function of the API; you should only need to use the predict endpoint in your solution.
 - Provide a very simple interface (e.g. read from STDIN, html page with search box, or GUI) that repeatedly reads in a string tag name and returns a sorted list of at most 10 of the most probable images.

# Solution: MVP level-1
The solution is implemented as three services, that could be run as separate processes on multiple machines, or within a single process.

![mvp1](img/mvp1.png)

1. Image tagging service
2. Search service
3. Search web server

This MVP level 1 solution is meant to run on a single server.

### Image tagging service
Classifies images and caches classification results on disk. Since calls to classification APIs are expensive, caching the results is required to minimize API calls to one per image.

Cached data is a map of image URL to array of tags.
```
URL -> [  
    {term, score},
    {term, score}, ...
]
```

### Search service
Cached classification results are loaded into memory and converted into a concordance.

```
term -> [
    {URL, score},
    {URL, score},...
]
```

### Search web server
A web interface allows users to enter a term to search for. The term is looked up against the corcodance to return matching results.

### Output
![sample results](img/results.png)

# Solution: MVP-2

### Limitation of MVP-1
Searching is limited to exact string matching of search terms. For example, a search for 'dwelling' would not retrieve results of houses.

### Word embeddings
Instead of matching terms exactly, we can convert classification results into word embeddings (vector of numbers, say 100 long) using libraries like word2vec.
User's search term too is mapped to a work embedding.

Word embeddings in the cache within a certain 'distance' of the search term's embedding would give us better results.

This works even if the exact search term is not present in the cache.
For example: searching for 'dwelling' would pull up 'bungalow', 'farmhouse', etc.

In this implementation, I used GloVe (Global Vectors for Word Representation) to measure word similarity.

### MVP-2 Output

Searching for "organ" returns "piano", "instrument", "violin", "flute", "flute", "pipe", "cello", "guitar" and "orchestra",

![GloVe output](img/results3.png)

Searching for "dessert" returns results like "delicious", "appetizer", "cake", "chocolate", "dish" and "pudding". 

![GloVe output](img/results4.png)


Searching for "dwelling" pulls up results like "bungalow", "farmhouse", "bedroom", "porch" and "rustic". 

![GloVe output](img/results2.png)


### Algorightm

Refer to search2/wordsim.go.

It is an adaptataion of code from https://github.com/ynqa/word-embedding

- Load word vectors from disk into memory
  ``` 
  corpus := map[word] -> tensor
  ```
- Extract vectors for tag-words in cache (from concordance)
  ```
  index := intersection(corpus & tags)
  ```
- For each search term, find its tensor from corpus
  ```
  target-tensor := corpus[term]
  ```
- Measure distance to each word in index using cosine measure - [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
  ```
  distance := pearson_coeff(target-tensor, index-tensor)
  ```
- Sort words by distance and return top 10

## Future improvements

### Improved word embeddings

- Choose better corpus and tune word vector length
- Select better distance measure for word vectors

- Using matrix math on GPUs, the distance calculations can be batched, returning sub-second results.

### Scaling up

The three services could be split up to run as distributed services on multiple servers.

![mvp2](img/mvp2.png)

Map-reduce could be used to aggregate results from multiple servers.

Publish-subscribe could be used to keep the services updated when new data becomes available.

# Installation and Usage
Pre-requisites
- Install [golang](https://golang.org/) version 1.10.3
- Install tensor package
  ``` 
   go get -u "github.com/chewxy/gorgonia/tensor"
   ```
- Download [pre-trained word vectors](http://nlp.stanford.edu/data/glove.6B.zip) of Wikipedia 2014 + Gigaword 5 dataset from the Stanford (GloVe project page)[https://nlp.stanford.edu/projects/glove/].
- Extract zip file to ./glove folder

**Executing Search (MPV-1)**

From terminal, run

 ```
 go run search/search.go -in ../data/cache.json
 ```

_Note: use argument -port nnnn to change default port (default: 80)_

Open browser to http://localhost/ and enter  a search term to view results.

**Executing Search (MPV-2)**

From terminal, run

 ```
 go run search/search.go -in ../data/cache.json -glove ../glove/glove.6B.300d.txt
 ```

**Running classification**

This step is necessary to recreate classification results file on disk.

From terminal, run

 ```
 go run classify/classify.go -in ../data/images.txt -out ../data/cache.json
 ```
