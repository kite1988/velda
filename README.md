# Visual Emotional Latent Dirichlet Allocation (VELDA) 

This is the implementation based on the following paper:

Tao Chen, Hany M. SalahEldeen, Xiangnan He, Min-Yen Kan and Dongyuan Lu (2015). [VELDA: Relating an Image Tweet’s Text and Images.](http://wing.comp.nus.edu.sg/~chentao/data/pubs/velda_aaai15.pdf) In Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI'15), Austin, USA.

 We additioanlly release the preprocessed [Wikipedia Picture of the Day](https://en.wikipedia.org/wiki/Wikipedia:Picture_of_the_day) (POTD) dataset. See [dataset/POTD] (https://github.com/kite1988/velda/tree/master/dataset/POTD) for details. 
 
**Please cite our AAAI'15 paper if you use our code or dataset. Thanks!** 

Author: Tao Chen (http://wing.comp.nus.edu.sg/~chentao)


## Usage

* Change the configuration in [config.init](https://github.com/kite1988/velda/blob/master/config.init)
* Run java command:

  ``` java Main.java config.init```
  
  If you are training with a large dataset, please allocate more memory to JVM, e.g.,
  
   ``` java -Xmx2g Main.java config.init```
  
  This code invokes the pipeline of training, testing and evaluation. The evaluation is conducted in a text-based image retrieval task. Please see the paper for detailed description.
   

## Dataset (input) format
  * Training set consists of three files:
    * p_train_text.txt
    * p_train_visual.txt
    * p_train_emotion.txt
  
  Each line contains the textual/visual/emotional word IDs (seperated by a whitespace) for a particular document. That is, the n*th* lines correspond to the n*th* document in the dataset. Note word IDs are continuous integers ( [0, vocabulary size-1])

  * Test set consists of three files:
    * p_test_text.txt
    * p_test_visual.txt
    * p_test_emotion.txt
  
  The file formats are similar to the training set.

  Please see [dataset/POTD] (https://github.com/kite1988/velda/tree/master/dataset/POTD) for an example dataset. 


## Output

  * Training (model)

    File | Description | Dimension
    ------------ | ------------- | -------------
    p_zt_t.txt | textual topic word distribution | (K+E) * T
    p_zv_t.txt | image-visual topic word distribution | K * C
    p_ze_t.txt | image-emotion topic word distribution | E * S
    p_t_r.txt  | textual word relevance distribution | T * 2
    p_d_zv.txt | document visual-topic distribution | D * K
    p_d_ze.txt | document emotion-topic distribution | D * E
    
    Notation: 
    * K/E is the number of visual/emotional topics.
    * T/C/S is the vocabulary size of textual/visual/emotional words.
    * D is the number of training documents.
  
  
  * Testing
  
    File | Description | Dimension
    ------------ | ------------- | -------------
    p_d_zv.txt | document visual-topic distribution | D * K
    p_d_ze.txt | document emotion-topic distribution | D * E

    Notation:
      * K/E is the number of visual/emotional topics.
      * D is the number of testing documents.
      
  * Evaluation
  
    * result.csv: error rate against different top percentage of retrieved results
    * top_results.csv: each line contains the top five results (image ID and score), along with grouth truth and a random image for a query text.
    
    Note the image IDs are continuous integers ([0, D-1]).











