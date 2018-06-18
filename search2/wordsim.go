package main

import (
	"bufio"
	"errors"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"gorgonia.org/tensor"
)

// Distance of word from target
type Distance struct {
	word string
	dist float64
}

// Distances is the list of Sim.
type Distances []Distance

func (m Distances) Len() int           { return len(m) }
func (m Distances) Less(i, j int) bool { return m[i].dist < m[j].dist }
func (m Distances) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }

// SimilarWords class returns words that are 'close' to a given target word
type SimilarWords struct {
	corpus map[string]*tensor.Dense
	index  map[string]*tensor.Dense
}

// Constructor to load GloVe data into memory
// TODO: make this a singleton
func NewSimilarWords(corpus string, words []string) *SimilarWords {
	var sw SimilarWords
	sw.corpus = loadWordVec(corpus)
	// fetch tensors for tag-words
	sw.index = make(map[string]*tensor.Dense)
	for _, w := range words {
		tvec, ok := sw.corpus[w]
		if !ok {
			continue // skip words not present
		}
		sw.index[w] = tvec
	}
	return &sw
}

// Return top 10 similar words
// TODO: instead of top-10, return thresholded words by %
// TODO: cache norms for index map
func (sw SimilarWords) GetSimilarWords(target string) ([]string, error) {

	tvec, ok := sw.corpus[target]
	if !ok {
		return nil, errors.New("word not found")
	}

	tvecNorm := norm(tvec)

	res := make(Distances, len(sw.index))

	for word, vec := range sw.index {
		if word == target {
			continue
		}

		dist := pearson_coeff(tvec, tvecNorm, vec)

		res = append(res, Distance{
			word: word,
			dist: dist,
		})
	}

	sort.Sort(sort.Reverse(res))

	result := make([]string, 10)
	for i := 0; i < 10 && i < len(res); i++ {
		result[i] = res[i].word
	}
	return result, nil
}

// loadWordVec read GloVe file into a map
func loadWordVec(fname string) map[string]*tensor.Dense {
	corpus := make(map[string]*tensor.Dense)

	f, err := os.Open(fname)
	if err != nil {
		log.Fatal(err)
		return nil
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, " ") {
			continue
		}
		word, vec, err := parse(line)
		if err != nil {
			log.Fatal(err)
			return nil
		}

		corpus[word] = vec
	}
	return corpus
}

// Parse each line from file
func parse(line string) (string, *tensor.Dense, error) {
	sep := strings.Fields(line)
	word := sep[0]
	v := sep[1:]
	vec := tensor.NewDense(tensor.Float64, tensor.Shape{len(v)})
	dat := vec.Data().([]float64)
	for k, elem := range v {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			log.Fatal(err)
			return "", nil, err
		}
		dat[k] = val
	}
	return word, vec, nil
}

// norm calculates the matrix p-norm
func norm(d *tensor.Dense) float64 {
	n, _ := d.Norm(tensor.UnorderedNorm())
	return n.ScalarValue().(float64)
}

// pearson_coeff distance measure
func pearson_coeff(d1 *tensor.Dense, d1Norm float64, d2 *tensor.Dense) float64 {
	inner, _ := tensor.Inner(d1, d2)
	return inner.(float64) / (d1Norm * norm(d2))
}
