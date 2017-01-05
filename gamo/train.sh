#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"

train ()
{
    if [ ! -f $2/model_1.t7 ]; then
        th main.lua -data $ALIGNED_DIR/train -modelDef $1 -cache $WORK_DIR/data/cache  \
            -save $2  -nDonkeys 8  -peoplePerBatch 7 -imagesPerPerson $4 -testing \
            -epochSize 1000 -nEpochs 250 -imgDim 64 -criterion $3 $5
    fi
}

cd ../training

for i in triplet loglikelihood kl
do
    MODEL=$WORK_DIR/../models/mine/nn4.small2.def.64_1.lua
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"
    if [ ! -d $RESULT_DIR ]; then

        train $MODEL $RESULT_DIR $i 30
    fi
done


for i in hinge
do
    MODEL=$WORK_DIR/../models/mine/nn4.small2.def.64_1_par.lua
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"
    if [ ! -d $RESULT_DIR ]; then

        train $MODEL $RESULT_DIR $i 30
    fi
done