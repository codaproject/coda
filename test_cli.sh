#!/bin/bash
source .env
export OPENAI_API_KEY
python -m coda.cli --text data/test_transcript.txt --grounder rag --agent toy --output data/output.txt