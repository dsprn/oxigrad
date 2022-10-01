# oxigrad

## What is it?
This is a partial port of [capmangrad](https://github.com/dsprn/capmangrad) to the Rust programming language, just to get a feeling of how the language works.
This implementation lacks some of the features capmangrad added on top of [micrograd](https://github.com/karpathy/micrograd). For a complete list see the Todos section below.

## How to use it
This project can be run from the command line with the following command (once positioned inside the project directory)
```
cargo run
```
Once run it will produce an output similar to the following one (choosing the best hyperparameter for L2 regularization within a given range and than training a model using it to counter overfitting)
```
==> Using Cross Validation to look for the best L2 lambda hyperparameter in values ranging from 0 to 0.01
hyperpar=0.0005, accuracy=52%
hyperpar=0.0010, accuracy=52%
hyperpar=0.0015, accuracy=40%
hyperpar=0.0020, accuracy=51%
hyperpar=0.0025, accuracy=57%
hyperpar=0.0030, accuracy=45%
hyperpar=0.0035, accuracy=52%
hyperpar=0.0040, accuracy=44%
hyperpar=0.0045, accuracy=58%
hyperpar=0.0050, accuracy=51%
hyperpar=0.0055, accuracy=50%
hyperpar=0.0060, accuracy=61%
hyperpar=0.0065, accuracy=49%
hyperpar=0.0070, accuracy=51%
hyperpar=0.0075, accuracy=49%
hyperpar=0.0080, accuracy=51%
hyperpar=0.0085, accuracy=53%
hyperpar=0.0090, accuracy=57%
hyperpar=0.0095, accuracy=45%
hyperpar=0.0100, accuracy=57%
==> L2 lambda value=0.0060

==> Choosing inputs and relative label from a preloaded dataset...
==> Input values=[1.89457429, 0.36178464]
==> Expected value=1

==> Start training the model...
pass=0, alpha=0.030, prediction=-0.679357, reg=0.630883, loss=2.820238, tot_loss=3.451121
pass=1, alpha=0.030, prediction=3.907141, reg=0.631138, loss=8.451470, tot_loss=9.082608
pass=2, alpha=0.029, prediction=-1.696822, reg=0.618529, loss=7.272851, tot_loss=7.891380
pass=3, alpha=0.029, prediction=0.286639, reg=0.612443, loss=0.508884, tot_loss=1.121327
pass=4, alpha=0.028, prediction=0.666767, reg=0.612471, loss=0.111044, tot_loss=0.723515
pass=5, alpha=0.028, prediction=0.814095, reg=0.612441, loss=0.034561, tot_loss=0.647002
pass=6, alpha=0.028, prediction=0.899078, reg=0.612287, loss=0.010185, tot_loss=0.622472
pass=7, alpha=0.027, prediction=0.945714, reg=0.612033, loss=0.002947, tot_loss=0.614980
pass=8, alpha=0.027, prediction=0.970530, reg=0.611718, loss=0.000868, tot_loss=0.612586
pass=9, alpha=0.026, prediction=0.983570, reg=0.611371, loss=0.000270, tot_loss=0.611641
pass=10, alpha=0.026, prediction=0.990417, reg=0.611009, loss=0.000092, tot_loss=0.611101
==> DONE

```

To compile the project run the following command in your terminal (this generates a dev executable, i.e. not optimized for production)
```
cargo build
```
or for a production ready executable use this instead
```
cargo build --release
```

## Tests
To run the tests associated with most the structures and methods present in this code type
```
cargo test
```
and you'll get an output with the tests results.

## Todos
Following are the features that are present in capmangrad but that are still missing in this version.
Listed here in no particular order:
* save model to a json file
* visualize the computational graph
