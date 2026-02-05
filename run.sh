#!/bin/bash

# if the first argument is undefined, set it to a default value
masterjson=${1:-"$HOME/tdata/master.json.gz"}
trimTracksOut=${2:-"output/output.json.gz"}
components=${3:-"2"}
n_neighbors=${4:-"50"}
n_epochs=${5:-"200"}

# if catnames-cli is on the PATH, use it otherwise use the full path
  

# if the trimTracksOut file does not exist, create it
if [ ! -f "$trimTracksOut" ]; then

# 👍 👍 👍 👍 👍 👍👍 👍 👍 👍 👍 👍👍 👍 👍 👍 👍 👍👍 👍 👍 👍 👍 👍 
# https://github.com/tidwall/gjson/blob/master/SYNTAX.md 👍 👍 👍 👍 👍 👍
# 👍 👍 👍 👍 👍 👍👍 👍 👍 👍 👍 👍👍 👍 👍 👍 👍 👍👍 👍 👍 👍 👍 👍
# catnames-cli: https://github.com/rotblauer/cattracks-names
# mac not like zcat zcat, need cat zcat
# ,#(properties.Activity!="")
  cat $masterjson|zcat \
  |catnames-cli modify --name-attribute 'properties.Name' --sanitize true \
  |go run main.go \
    --match-all '#(properties.Speed<50),#(properties.Accuracy<10)' \
    --match-any '#(properties.Name=="ia"),#(properties.Name=="rye")' \
    filter \
    |gzip  > $trimTracksOut
    # ,#(properties.Activity!="unknown")
    
else
    echo "File $trimTracksOut already exists"
    
fi

# run json_to_tsv.py on the trimTracksOut file
# use awk to select every 10th line for sampling

cat $trimTracksOut \
|zcat \
|.venv/bin/python json_to_tsv.py \
--output "output/raw.tsv.gz"
