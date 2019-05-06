# Basic MNIST Example

**Questions**:

* Can we fragment a model and give a piece to the client such that they encode their data in
a non-trivial way specifically relevant to the task that a service provider wishes to perform on that
data?
* How hard would it be for a service provider to re-create the original sample?  

This example makes use of the mnist classification task to demonstrate the ability to split a model
and the efficacy with which a service provider can reconstruct original samples.

## Install and Run example

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

## Split model

After training, the model is split such that the weights of the frontend piece can be transmitted to
the client, and the secondary piece then operates on the generated intermediary.

This works naturally as there is no change in the weights or the dimensionality of the model.

## Reverse Model

As we rely on the intermediary representation to provide some measure of obfuscation we next test
the reliability with which the model can be reversed by the service provider to re-create the original
sample.

This is done simply as we have both the orioginal as a label and the generated intermediary to test against.

## Discussion

### Benefits

Clients don't receive the entire model so there is still some ides of a proprietary model.

We offload some of the computation to the client so that they can gain this benefit decreasing
the computational load on the server.

### Drawbacks

We ofload some work to the client increasing the amount of work that has to be done on
non-specialized hardware.

## Question for future work

* How do model complexity and the distance through the model where the split is inserted affect:
    - the amount of computational work a client has to do
    - the ability of a service to re-create the origianl sample
    - the compression factor of the sample for network transfer
    - the size of required model weight initializations transfered to the client

* Can we construct the client side model fragment to be transferable to related classification tasks
without directly revealing information about the client?