#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
ngram -lm /root/pfr-work/pfr6/models/kn3.arpa -ppl /root/pfr-work/pfr6/splits_main/test.txt -unk 2>&1 | head -20
