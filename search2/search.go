package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"sort"
)

var templ = template.Must(template.New("ImageSearch").Parse(templateStr))

// Tag represents a single match
type Tag struct {
	Name  string
	Score float32
}

// Matches - typedef for Tag[]
type Matches []Tag

// Dict - map<term,Matches>
type Dict map[string]Matches

// Functions to desceding sort Matches
func (m Matches) Len() int {
	return len(m)
}
func (m Matches) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}
func (m Matches) Less(i, j int) bool {
	// reverse sort
	return m[i].Score > m[j].Score
}

// load deserializeds map<url, Tag[]> from disk
// TODO: stream for larger files
func load(fname string) Dict {
	buf, err := ioutil.ReadFile(fname)
	if err != nil {
		log.Fatal(err)
	}
	v := Dict{}
	if err := json.Unmarshal(buf, &v); err != nil {
		log.Fatal(err)
	}
	return v
}

// makeConcordance flips map<url, term[]> to map<term, url[]> for quick lookup
// TODO: merge with load for single-step load
func makeConcordance(cache Dict) Dict {
	conc := make(Dict)
	for url, tags := range cache {
		for _, tag := range tags {
			c := conc[tag.Name]
			if c == nil {
				c = make([]Tag, 0)
			}
			c = append(c, Tag{Name: url, Score: tag.Score})
			conc[tag.Name] = c
		}
	}
	return conc
}

// MatchSet pairs term and results
type MatchSet struct {
	Term    string
	Matches Matches
}

// searchImages looksup concordance and returs sorted top 10
func searchImages(sim SimilarWords, concordance Dict, term string) []MatchSet {

	terms := []string{term}
	similarWords, err := sim.GetSimilarWords(term)
	if err == nil {
		terms = append(terms, similarWords...)
	}

	var matchSets []MatchSet
	for _, t := range terms {
		matches := concordance[t]
		sort.Sort(matches)
		if len(matches) > 10 {
			matches = matches[:10]
		}
		matchSets = append(matchSets, MatchSet{
			Term:    t,
			Matches: matches,
		})
	}

	return matchSets
}

// Port argument - default 80 : ana_d
var port = flag.Int("port", 80, "http service port")
var inputFile = flag.String("in", "../data/cache.json", "Cache file to use")
var gloveFile = flag.String("glove", "../glove/glove.6B.300d.txt", "Location of pre-trained glove word vector file")

func main() {

	flag.Parse()
	cache := load(*inputFile)
	fmt.Printf("Loaded %d URLs\n", len(cache))
	concordance := makeConcordance(cache)
	fmt.Printf("Loaded %d terms\n", len(concordance))

	var words []string
	for k := range concordance {
		words = append(words, k)
	}
	sim := NewSimilarWords(*gloveFile, words)
	fmt.Println("Loaded corpus")

	// closure - capture concordance in callback
	fn := func(w http.ResponseWriter, req *http.Request) {
		requestHandler(*sim, concordance, w, req)
	}

	// fire up http server
	http.Handle("/", http.HandlerFunc(fn))
	err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
	if err != nil {
		log.Fatal("ListenAndServe:", err)
	}
}

// Result of image search
type Result struct {
	Valid      bool
	SearchTerm string
	MatchSets  []MatchSet
}

// requestHandler executes search
func requestHandler(sim SimilarWords, concordance Dict, w http.ResponseWriter, req *http.Request) {

	searchTerm := req.FormValue("searchTerm")
	var result Result
	if searchTerm != "" {
		matchSets := searchImages(sim, concordance, searchTerm)
		result = Result{
			Valid:      true,
			SearchTerm: searchTerm,
			MatchSets:  matchSets,
		}
	}
	templ.Execute(w, result)
}

// HTML template
const templateStr = `
<html>
<head>
<title>Image search</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" 
	integrity="sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB" crossorigin="anonymous">
</head>
<body class="text-center">
<h1 class="h3 mb-3 font-weight-normal">Image Search</h1>
<div class="w-75 p-3 container-fluid">
<form action="/" name=f method="GET">
	<div class="row justify-content-between">
		<input class="col-9 form-control" name="searchTerm" placeholder="Enter search term">
		<button class="col-2 btn btn-primary" type="submit">Search</button>
	</div>
</form>
{{if .Valid}}
{{if .MatchSets}}
	{{range .MatchSets}}
		{{if .Matches}}
			<p class="h5 mb-3 font-weight-normal">Search results for: {{.Term}}</p>
			<div class='d-flex align-content-around flex-wrap'>
				{{range .Matches}}
					<div class='p-2 bd-highlight'>
						<img src="{{.Name}}" width=100 />
						<p>{{.Score}}</p>
					</div>
				{{end}}
			</div>
		{{else}}
			<p class="h5 mb-3 font-weight-normal">No matches for {{.Term}}</p>
		{{end}}
	{{end}}
{{else}}
	<p class="h5 mb-3 font-weight-normal">No matches for {{.SearchTerm}}</p>
{{end}}
{{end}}

</div>
</body>
</html>
`
