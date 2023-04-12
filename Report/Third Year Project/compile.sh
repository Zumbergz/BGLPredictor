#!/bin/bash

OUTDIR=./PDF
FILENAME=main

OPTIONS="--synctex=1 -interaction=nonstopmode"

mkdir -p $OUTDIR

lualatex $OPTIONS --output-directory=$OUTDIR $FILENAME.tex
biber --output-directory=$OUTDIR $FILENAME
lualatex $OPTIONS --output-directory=$OUTDIR $FILENAME.tex
lualatex $OPTIONS --output-directory=$OUTDIR $FILENAME.tex
cp $OUTDIR/$FILENAME.pdf .