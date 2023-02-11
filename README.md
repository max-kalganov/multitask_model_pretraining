# Multitask model pretraining

The aim of this project is to test ability of the model
to extract common preprocessing operations for classifying
different classes.

This is implemented via custom one-vs-all approach. Model
contains single input and a few base layers, which are continued
by several heads for classifying different classes.
Each head has a binary output classifying input containg/not-containing
a specific class.

The experiment is tested on ImageNet dataset.


### Iterations

1. Make a single model for all classes and store metrics
2. Make a single model with a common base, but different heads for each class
3. Make different models with for each class with pretrained base from 1. and then from 2. 

4. Use pretrained base for classifying unseen class, but the base remain frozen

### Expectations

Training a new class would require less layers archiving the same result
using frozen base lsyers.
