LIBLINEAR_bitmap
=============

```
Usage: train [options] training_set_file [model_file]

options:

-s type : set type of solver (default 1)
	1 -- L2-regularized L2-loss support vector classification (dual)
	3 -- L2-regularized L1-loss support vector classification (dual)
-c cost : set the parameter C (default 1)
-d cost : set the parameter C2 (default 1) Usable only with '-m 3'
-e epsilon : set tolerance of termination criterion
-B bias : if bias = 1, instance x becomes [x; 1]; if < 0, no bias term added (default -1)
-wk weight: weights adjust the parameter C of different classes (default: -wk=1 for all k-th class)
-q : quiet mode (no outputs)
-W weight_file: set weight file. (default: W[i]=1, for all i)
-m mode : 1-vs-all mode (default 0)
    0 default LIBLINEAR behaviour
    1 Cpos=C/num_pos, Cneg=C/num_neg   (and every slack is also multiplied for W[i])
    2 Cpos=(C*num_neg)/num_pos, Cneg=C    (and every slack is also multiplied for W[i])
    3 Cpos=C, Cneg=C2    (and every slack is also multiplied for W[i])
-t num_threads   (usable only in the 1-vs-all mode)
-n num_neg_per_class (default 0, i.e. all the examples. Usable only in the 1vsAll mode)
-r mode_neg_set_selection This options applies only if -n >0.
    If 0 (default) we randomly select the negative set for each class
    if 1 we deterministicly select the first '-n' examples in order.
       Note: In this case it loads only the data strictly necessary for the learning.
             If the data is divided in chunks each has the examples of a label, that is translated in a significant speed-up during the loading
-l <list of index classes separated by colums e.g 0,1,2,3,7> 
   This computes only the specified 1vsAll and save a file for each model <model_file>_cl%d
   Note: you don't have to specify the labels but the index e.g. idx \in 0:length(unique(tr_label))
-p <fileName> Apply pairwise products to the data. We specify the products in a file (It is an ASCII file containing a float matrix [D x 2] of int. The index of the pairwise start from zero)
-b <b>:<k> Apply b-bit Minwise hashing to the data. N.B.: mod(b*k, 8)==0 and b<=32
-f <command> Command to execute right before the data of the loading
-g <command> Command to execute right after the data of the loading


Note:
In the 2-class case, the weights of the slack variables are computed in the following way:
- Binary problem: For the example i-th, belonging to the class k-th, the weight is: C*wk*W[i]
- 1-vs-all: for each class k-th, the positive points have wk*W[i], the negative C*W[i]

Note:
The output between -t 1 and -t >1 may vary slightly (due to a random example selection during the optimization) 
```
