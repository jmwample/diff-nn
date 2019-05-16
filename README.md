# Just the Facts Please

Conv2d => Conv2d

Client segnment of the model involves Conv2d operation.

The reverse model uses [maxpool, conv2d, deconv2d, maxunpool, deconv2d] layers to reconstruct the original image.

![Framework](https://raw.githubusercontent.com/jmwample/diff-nn/master/img/diffnn_framework.png)

**(Step 1)** involves the __design and training of a model__ -- part of which
(*step 1a.*) requires choosing the layers that will be transferred to the client. For
***pre-trained*** models step 1 is complete given that there is a satisfactory
point at which to segment the network.

**(Step 2)** __splits the network__ into two pieces, the first (client) portion
is transferred to the client, the second is maintained locally.


**(Step 3)** __evaluates the invertibility__ of the primary layer we construct
a new modeland train it on the original training data where we pass an input through
the client segment,then pass the intermediate through the reconstruction network and
calculate lossagainst the original input value. Loss is propagated only through the
reconstruction networkas we ***do not*** wish to optimize the client model segment
for reconstruction.

## Basic MNIST Example

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

### Result

MLP Client => MLP Reconstruction:

![](https://raw.githubusercontent.com/jmwample/diff-nn/master/img/reconstruction_10_ll.png)

Conv2d Client => MLP Reconstruction:

![](https://raw.githubusercontent.com/jmwample/diff-nn/master/img/reconstruction_10_cl.png)

Conv2d Client => [maxpool, conv2d, deconv2d, maxunpool, deconv2d] Reconstruction:

![](https://raw.githubusercontent.com/jmwample/diff-nn/master/img/reconstruction_10_cc.png)


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

* How hard is it to undo operations like conv / Maxpool without indices?
