# oxigrad

## What it is?
This is a partial port of [capmangrad](https://github.com/dsprn/capmangrad) to the Rust programming language, just to get a feeling of how the language works.
This implementation lacks most of the features capmangrad added on top of [micrograd](https://github.com/karpathy/micrograd). For a complete list see the todos below.

## How to use it
This project can be run from the command line with the following command (once positioned in the project dir)
```
cargo run
```
It will produce an output similar to the following one (proving it can traing a model, even if it's an overfitted one)
```
==> Expected value=2.314
==> Start training the model...
epoch:1, prediction:-0.009883676, loss:5.4004345
epoch:2, prediction:0.14083661, loss:4.722638
epoch:3, prediction:0.28066546, loss:4.134449
epoch:4, prediction:0.41289008, loss:3.6142185
epoch:5, prediction:0.5397332, loss:3.1480224
epoch:6, prediction:0.6630315, loss:2.7256963
epoch:7, prediction:0.7845003, loss:2.3393688
epoch:8, prediction:0.905812, loss:1.9829931
epoch:9, prediction:1.0285573, loss:1.6523627
epoch:10, prediction:1.1540942, loss:1.3453811
epoch:11, prediction:1.2832744, loss:1.062395
epoch:12, prediction:1.4160447, loss:0.8063235
epoch:13, prediction:1.5509899, loss:0.5821843
epoch:14, prediction:1.6849957, loss:0.39564633
epoch:15, prediction:1.8133404, loss:0.2506599
epoch:16, prediction:1.9304817, loss:0.14708622
epoch:17, prediction:2.0314722, loss:0.07982189
epoch:18, prediction:2.1134028, loss:0.040239174
epoch:19, prediction:2.1760755, loss:0.019023148
epoch:20, prediction:2.2215948, loss:0.008538699
epoch:21, prediction:2.2532918, loss:0.003685467
epoch:22, prediction:2.2746716, loss:0.0015467181
epoch:23, prediction:2.288766, loss:0.000636754
epoch:24, prediction:2.297913, loss:0.00025878567
epoch:25, prediction:2.3037877, loss:0.00010428868
epoch:26, prediction:2.3075345, loss:0.00004180185
epoch:27, prediction:2.3099136, loss:0.000016697488
epoch:28, prediction:2.3114202, loss:0.0000066547955
epoch:29, prediction:2.3123724, loss:0.000002648578
==> DONE
```

To compile the project run the following in your terminal (for a not so optimized dev executable)
```
cargo build
```
or (for a production ready executable)
```
cargo build --release
```

Right now each model hyperparameter is hard-coded and there's no way to pass the executable any paramters.

## Tests
To run the tests associated with most the structures and their methods present in this code type
```
cargo test
```
and you'll get an output with the results.

## Todos
Following are the features that were present in capmangrad but that are missing in this version.
Listed here in a random order:
* more loss functions
* L2 regularization
* cross validation
* saving model to file
* computational graph visualization
