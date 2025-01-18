#!/bin/bash

# Define the file name
filename="timestamps.txt"

# Check if the file exists
if [ -f "$filename" ]; then
    sed -i '1d' "$filename"
    echo "First line of $filename deleted successfully."
else
    echo "Error: File $filename not found."
fi