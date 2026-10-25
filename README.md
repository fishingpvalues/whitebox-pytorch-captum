# whitebox-pytorch-captum

A notebook (`captum.ipynb`) that trains a small PyTorch classifier and explains it with
[Captum](https://captum.ai/) attribution methods.

## What it does

- Data: scikit-learn's breast cancer dataset (30 numeric features, binary label),
  standardised, 70/30 train/test split.
- Model: a feed-forward network, `Linear(30, 64)`, ReLU, `Linear(64, 1)`, sigmoid;
  binary cross-entropy loss, Adam with learning rate 0.001, 100 epochs, full batch.
- Attribution: every 10 epochs it computes attributions for the training set with
  Integrated Gradients, Saliency, DeepLift, Input x Gradient and GradientShap (random
  normal baselines). For the final epoch it plots the mean attribution per feature for
  each method as grouped bars, next to the normalised weights of the first linear unit
  (grey) for comparison.
- Evaluation: test accuracy 0.9825, ROC AUC 0.9988, classification report, confusion
  matrix and ROC curve (as recorded in the notebook output).
- Architecture graph rendered with `torchviz`.

The last cell repeats the setup with 200 epochs and without the evaluation code; it
is an unfinished variant.

## Run

```bash
pip install torch captum scikit-learn matplotlib seaborn numpy torchviz
jupyter notebook captum.ipynb
```

`torchviz` also needs the Graphviz binaries.

## Scope

This is an illustration of the Captum API on a small tabular problem. The notebook
compares methods visually only; it does not quantify attribution quality, and the
model is trained on the full training set without validation.
