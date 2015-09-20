#!/bin/bash
# Converts rst files to markdown using Pandocs

RST_FILES=../doc/*.rst

for f in $RST_FILES 
do
    echo "Processing $f"
    filename=$(basename "$f");
    filename="${filename%.*}";
    pandoc $f -f rst -t markdown -o "./docs/$filename.md"
done


