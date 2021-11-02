#!/bin/bash

INPUT="0";
OUTPUT="0";

while getopts hf:o: option; do
  case "${option}" in
    f) INPUT=${OPTARG};;
    o) OUTPUT=${OPTARG};;
    h)
       echo "";
       echo "$0: options";
       echo " -f input file, in trace format (1:window 4:peerid)";
       echo " -o output file, in snapshot format #peerid #window";
       echo "";
       exit 0;;
  esac;
done

if [ $INPUT = "0" ]; then
  echo "-f: input trace file not specified.";
  exit 1;
fi;

if [ $OUTPUT = "0" ]; then
  echo "-o: output snapshot file name undefined.";
  exit 1;
fi;

awk '{print $1 " " $2 " " $4 " " $4 " " 1 " " 1}' $INPUT | sort -u -k1n,1 > $OUTPUT;


