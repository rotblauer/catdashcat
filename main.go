package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	catnames "github.com/rotblauer/cattracks-names"
	"github.com/tidwall/gjson"
)

type T struct {
	Type     string `json:"type"`
	Id       int    `json:"id"`
	Geometry struct {
		Type        string    `json:"type"`
		Coordinates []float64 `json:"coordinates"`
	} `json:"geometry"`
	Properties struct {
		Accuracy  float64   `json:"Accuracy"`
		Activity  string    `json:"Activity"`
		Elevation float64   `json:"Elevation"`
		Heading   float64   `json:"Heading"`
		Name      string    `json:"Name"`
		Notes     string    `json:"Notes"`
		Pressure  float64   `json:"Pressure"`
		Speed     float64   `json:"Speed"`
		Time      time.Time `json:"Time"`
		UUID      string    `json:"UUID"`
		UnixTime  int       `json:"UnixTime"`
		Version   string    `json:"Version"`
	} `json:"properties"`
}

func passesAccuracy(accuracy float64, requiredAccuracy float64) bool {
	return (accuracy > 0 && accuracy < requiredAccuracy) || requiredAccuracy < 0
}

func validActivity(activity string, require bool, removeUnknownActivity bool) bool {
	return !require || (activity != "" && (!removeUnknownActivity || activity != "Unknown"))
}

func parseStreamPerProperty(reader io.Reader, writer io.Writer, n int, property string, names map[string]bool, accuracy float64, requireActivity bool, removeUnknownActivity bool) {
	breader := bufio.NewReaderSize(reader, 1024*1024) // 1MB buffer
	bwriter := bufio.NewWriterSize(writer, 1024*1024) // 1MB buffer
	defer bwriter.Flush()

	dec := json.NewDecoder(breader)
	enc := json.NewEncoder(bwriter)

	m := make(map[string]int)
	pCount := 0
	totalCount := 0
	start := time.Now()
	for {
		var t T
		if err := dec.Decode(&t); err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		totalCount++
		//	switch on the property to select on
		switch property {
		case "Name":
			t.Properties.Name = catnames.SanitizeName(catnames.AliasOrName(t.Properties.Name))
			// if names is empty or contains the name, increment the count
			if passName(names, t) && passesAccuracy(t.Properties.Accuracy, accuracy) && validActivity(t.Properties.Activity, requireActivity, removeUnknownActivity) {
				m[t.Properties.Name]++
				if m[t.Properties.Name]%n == 0 {
					enc.Encode(t)
					pCount++
				}
			}
		default:
			panic("invalid property")
		}
		// Progress reporting
		if totalCount%100000 == 0 {
			elapsed := time.Since(start).Seconds()
			rate := float64(totalCount) / elapsed
			fmt.Fprintf(os.Stderr, "\r[parse] %dk processed, %dk output, %.0f/sec",
				totalCount/1000, pCount/1000, rate)
		}
	}
	elapsed := time.Since(start).Seconds()
	fmt.Fprintf(os.Stderr, "\n[parse] Done: %d processed, %d output in %.1fs\n", totalCount, pCount, elapsed)
	printMap(m, os.Stderr)
}

func passName(names map[string]bool, t T) bool {
	return len(names) == 0 || names[t.Properties.Name]
}

func printT(t T, w io.Writer) {
	enc := json.NewEncoder(w)
	err := enc.Encode(t)
	if err != nil {
		panic(err)
	}
}

func printMap(m map[string]int, w io.Writer) {
	enc := json.NewEncoder(w)
	err := enc.Encode(m)
	if err != nil {
		panic(err)
	}
}

var flagNumber = flag.Int("n", 100, "select every nth")
var flagProperty = flag.String("p", "Name", "property to select on - select every nth within unique values of this property")
var flagNames = flag.String("names", "", "names to select on")
var flagRequiredAccuracy = flag.Float64("min-accuracy", 100, "minimum accuracy to select on, set to -1 to skip")
var flagRequireActivity = flag.Bool("require-activity", true, "require a valid activity (non-empty)")
var flagRemoveUnknownActivity = flag.Bool("remove-unknown-activity", true, "remove unknown activity")

// example usage:
// cat /tmp/2019-01-*.json | go run main.go -n 100 -p Name -names kk,ia,rye,pr,jr,ric,mat,jlc -min-accuracy 100 -require-activity=true > /tmp/2019-01-uniq.json

func main() {
	flag.Parse()

	// parse the flagNames into a set

	if len(flag.Args()) > 0 && flag.Args()[0] == "filter" {
		filterStream(os.Stdin, os.Stdout, splitFlagStringSlice(*flagMatchAll), splitFlagStringSlice(*flagMatchAny), splitFlagStringSlice(*flagMatchNone))
		return
	}

	names := make(map[string]bool)

	// if the flagNames is not empty, split on comma and add to the set
	if *flagNames != "" {
		for _, name := range regexp.MustCompile(`,`).Split(*flagNames, -1) {
			names[name] = true
		}
	}

	parseStreamPerProperty(os.Stdin, os.Stdout, *flagNumber, *flagProperty, names, *flagRequiredAccuracy, *flagRequireActivity, *flagRemoveUnknownActivity)
}

var flagMatchAll = flag.String("match-all", "", "match all of these properties (gjson syntax, comma separated queries)")
var flagMatchAny = flag.String("match-any", "", "match any of these properties (gjson syntax, comma separated queries)")
var flagMatchNone = flag.String("match-none", "", "match none of these properties (gjson syntax, comma separated queries)")
var errInvalidMatchAll = errors.New("invalid match-all")
var errInvalidMatchAny = errors.New("invalid match-any")
var errInvalidMatchNone = errors.New("invalid match-none")

func filterStream(reader io.Reader, writer io.Writer, matchAll []string, matchAny []string, matchNone []string) {
	breader := bufio.NewReaderSize(reader, 1024*1024)  // 1MB buffer
	bwriter := bufio.NewWriterSize(writer, 1024*1024)  // 1MB buffer
	defer bwriter.Flush()

	count := 0
	passed := 0
	start := time.Now()
readLoop:
	for {
		read, err := breader.ReadBytes('\n')
		if err != nil {
			if errors.Is(err, os.ErrClosed) || errors.Is(err, io.EOF) {
				break
			}
			log.Fatalln(err)
		}
		count++
		if err := filter(read, matchAll, matchAny, matchNone); err != nil {
			continue readLoop
		}
		bwriter.Write(read)
		passed++
		// Progress and flush periodically
		if count%100000 == 0 {
			elapsed := time.Since(start).Seconds()
			rate := float64(count) / elapsed
			fmt.Fprintf(os.Stderr, "\r[filter] %dk processed, %dk passed (%.1f%%), %.0f/sec",
				count/1000, passed/1000, 100*float64(passed)/float64(count), rate)
			bwriter.Flush()
		}
	}
	elapsed := time.Since(start).Seconds()
	fmt.Fprintf(os.Stderr, "\n[filter] Done: %d processed, %d passed (%.1f%%) in %.1fs\n",
		count, passed, 100*float64(passed)/float64(count), elapsed)
}

// filter filters some read line on the matchAll, matchAny, and matchNone queries.
// These queries should be written in GJSON query syntax.
// https://github.com/tidwall/gjson/blob/master/SYNTAX.md
func filter(read []byte, matchAll []string, matchAny []string, matchNone []string) error {

	// Here we hack the line into an array containing only this datapoint.
	// This allows us to use the GJSON query syntax, which is designed for use with arrays, not single objects.
	if !gjson.ParseBytes(read).IsArray() {
		read = []byte(fmt.Sprintf("[%s]", string(read)))
	}

	for _, query := range matchAll {
		if res := gjson.GetBytes(read, query); !res.Exists() {
			return fmt.Errorf("%w: %s", errInvalidMatchAll, query)
		}
	}

	didMatchAny := len(matchAny) == 0
	for _, query := range matchAny {
		if gjson.GetBytes(read, query).Exists() {
			didMatchAny = true
			break
		}
	}
	if !didMatchAny {
		return fmt.Errorf("%w: %s", errInvalidMatchAny, matchAny)
	}

	for _, query := range matchNone {
		if gjson.GetBytes(read, query).Exists() {
			return fmt.Errorf("%w: %s", errInvalidMatchNone, query)
		}
	}
	return nil
}

func splitFlagStringSlice(s string) []string {
	if s == "" {
		return []string{}
	}
	return strings.Split(s, ",")
}
