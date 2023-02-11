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
