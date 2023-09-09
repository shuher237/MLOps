#!/bin/bash

while getopts "f:i:h" option; do
    case $option in
        h)
            echo "options:"
            echo "f: path to python scripts"
            echo "i: path to interpreter"
            exit;;
        f) folder=${OPTARG};;
        i) interpreter=${OPTARG};;
        \?)
            echo "Error: Invalid option"
            exit;;
    esac
done

cd $folder
$interpreter data_creation.py 
#$interpreter data_preprocessing.py 
#$interpreter model_preparation.py
#$interpreter model_testing.py