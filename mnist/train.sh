#!/usr/bin/env bash

imgDim=28
WORK_DIR=$PWD
DATA_DIR="raw"
EXTERNAL_DIR="/home/cenk/Documents/openface/losses"

train ()
{
    if [ ! -d $RESULT_DIR ]; then

        th main.lua -data /media/cenk/2TB1/alter_siamese/data/mnist/train -modelDef $1 -cache $WORK_DIR/data/cache${imgDim}  \
            -save $2  -nDonkeys 8  -peoplePerBatch 10 -imagesPerPerson $4 -testBatchSize 10  -testDir /media/cenk/2TB1/alter_siamese/data/mnist/train \
            -epochSize 600 -nEpochs 40 -imgDim $imgDim -criterion $3 -embSize $embSize

    fi
}

cd ../training

for MODEL_NAME in  alexnet #densenet tinynet
do
   for i in   s_cosine  t_orj  s_hadsell
    do
        for embSize in 128
        do
            MODEL=$WORK_DIR/../models/mine/$imgDim/$MODEL_NAME.def.lua
            RESULT_DIR="$EXTERNAL_DIR/results/mnist/${DATA_DIR}_${embSize}/${i}/$MODEL_NAME"
            # model_path, result_path, cost_function, imagePerPerson
            train $MODEL $RESULT_DIR $i 10

            #sh $WORK_DIR/test.sh
        done

    done
done

