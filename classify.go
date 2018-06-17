package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
)

const apiKey = "Key d65fa3310b85456f806ec63569eb420b"
const generalModelPredict = "https://api.clarifai.com/v2/models/aaa03c23b3724a16a56b629203edc62c/outputs"
const enumModels = "https://api.clarifai.com/v2/models"

// CAIClient is a Go client for clarifai's REST API
type CAIClient struct {
	client http.Client
}

// invoke exectues a REST call with dev. code in header
// returns json response as map[string] -> object
func (cai CAIClient) invoke(method string, url string, msg *string) interface{} {
	req, _ := http.NewRequest(method, url, strings.NewReader(*msg))
	req.Header.Add("Authorization", apiKey)
	req.Header.Add("Content-Type", "application/json; charset=UTF-8")
	resp, err := cai.client.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	var result interface{}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	json.Unmarshal(body, &result)
	return result
}

// EnumModels lists available classification models
func (cai CAIClient) EnumModels() interface{} {
	return cai.invoke("GET", enumModels, nil)
}

// PredictImage runs classification on one image
// TODO: batch multiple images per call ?
// TODO: serialize to json, escape url
func (cai CAIClient) PredictImage(url string) interface{} {
	msg := fmt.Sprintf(`{
		"inputs": [
		  {
			"data": {
			  "image": {
				"url": "%s"
			  }
			}
		  }
		]
	  }`, url)
	return cai.invoke("POST", generalModelPredict, &msg)
}

// Tag is a single classification result
type Tag struct {
	Name  string
	Score float32
}

// ImageClassifier class saves classification results to disk
// TODO: stream results to disk instead of caching in memory
type ImageClassifier struct {
	cai   CAIClient
	cache map[string][]Tag
}

// NewImageClassifier - constructor allocats cache
func NewImageClassifier() *ImageClassifier {
	var imgProc ImageClassifier
	imgProc.cache = make(map[string][]Tag)
	return &imgProc
}

// cast object to map<string,object>
func asMap(o interface{}) map[string]interface{} {
	return o.(map[string]interface{})
}

// ProcessImage classifies one image using clarifai client and updates cache
// return false on any failures
// TODO: batch multiple images for each call
// TODO: handle other error codes https://clarifai.com/developer/status-codes/
func (imgProc ImageClassifier) ProcessImage(url string) bool {
	fmt.Printf("Processing %s\n", url)
	result := asMap(imgProc.cai.PredictImage(url))
	code := asMap(result["status"])["code"].(float64) // parses as float64!
	if code == 10000.0 {                              // OK
		var tags = []Tag{}
		outputs := result["outputs"].([]interface{})
		data := asMap(asMap(outputs[0])["data"])
		concepts := data["concepts"].([]interface{})
		for _, v := range concepts {
			item := asMap(v)
			var tag Tag
			tag.Name = item["name"].(string)
			tag.Score = float32(item["value"].(float64))
			tags = append(tags, tag)
		}
		imgProc.cache[url] = tags
		return true
	}
	return false
}

// SaveCache serializes classification results to disk
func (imgProc ImageClassifier) SaveCache(fname string) int {
	outf, err := os.Create(fname)
	if err != nil {
		log.Fatal(err)
	}
	defer outf.Close()

	buf, err := json.Marshal(imgProc.cache)
	if err != nil {
		log.Fatal(err)
	}
	n, _ := outf.Write(buf)
	fmt.Printf("Wrote out %d\n", n)
	return n
}

// ProcessURLFile iterates through urls in file, classifying each
func (imgProc ImageClassifier) ProcessURLFile(fname string) {
	imageFile, err := os.Open(fname)
	if err != nil {
		log.Fatal(err)
		os.Exit(-1)
	}
	defer imageFile.Close()

	rdr := bufio.NewReader(imageFile)
	for line, err := rdr.ReadString('\n'); err == nil; line, err = rdr.ReadString('\n') {
		if !imgProc.ProcessImage(strings.TrimSpace(line)) {
			break
		}
	}
}

var inputFile = flag.String("in", "../images.txt", "Input file containing urls")
var outputFile = flag.String("out", "./cache.json", "file with classification results")

func main() {
	flag.Parse()
	imgCl := NewImageClassifier()
	imgCl.ProcessURLFile(*inputFile)
	imgCl.SaveCache(*outputFile)
}
