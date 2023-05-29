# Multi-Feature and Multi-Channel GCNs for Aspect Based Sentiment Analysisk

This repository contains the code for the paper "Multi-Feature and Multi-Channel GCNs for Aspect Based Sentiment Analysis".  
Please cite our paper and kindly give a star for this repository if you use this code.
## Setup

This code runs Python 3.7.0 with the following libraries:

+ Pytorch  1.13.1+cu116
+ Transformers 2.9.1
+ spacy 2.0.18

## Get start

1. Prepare data

   + Restaurants, Laptop, Tweets and MAMS dataset.

   + Downloading Glove embeddings (available at [here](http://nlp.stanford.edu/data/glove.840B.300d.zip)), then  run 

     ```
     awk '{print $1}' glove.840B.300d.txt > glove_words.txt
     ```

     to get `glove_words.txt`.
     

2. Build vocabulary

   ```
   bash build_vocab.sh
   ```
3. Build aspect-graph and inter-graph  
    + Go to the common folder:  
    
    + Generate aspect-focused graph with 
      
    ```
    python focused_graph.py
    ```
    
    + Generate inter-aspect graph with 
  
    ```
    python inter_graph.py
    ```
    
4. Training

   Go to Corresponding directory and run scripts:

   ``` 
   bash run-MAMS-glove.sh
   bash run-MAMS-BERT.sh
   ```

5. The saved model and training logs will be stored at directory `saved_models`  


6. Evaluating trained models (optional)

   ``` 
   bash eval.sh path/to/check_point path/to/dataset
   bash eval-BERT.sh path/to/check_point path/to/dataset
   ```
7. Notice 
Please remove the comments in the code to adapt it to different datasets.


##  Credits
The code of this repository partly relies on [InterGCN](https://github.com/BinLiang-NLP/InterGCN-ABSA) & [RGAT](https://github.com/goodbai-nlp/RGAT-ABSA/tree/master) & [DM-GCN](https://github.com/pangsg/DM-GCN).  
I would like to extend my appreciation to the authors of the InterGCN, RGAT, and DMGCN repositories for their valuable contributions.
