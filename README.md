# llm-fundamentals

Hands-on notes and code while working through language-model fundamentals from
scratch, in the spirit of Andrej Karpathy's *makemore*. Dataset: `names.txt`
(32K English first names, one per line); the task is character-level name
generation.

## Contents

- `makemore.ipynb` — count-based bigram model (27x27 count matrix, add-one
  smoothing, sampling), then the same bigram model trained as a single linear
  layer with gradient descent on negative log-likelihood plus L2 regularization.
- `trigram_splits.py` — count-based bigram and trigram models with an 80/10/10
  train/dev/test split, add-alpha smoothing selected on the dev set, and
  negative log-likelihood evaluation; the test set is touched once.
  Result: trigram test NLL 2.22 vs bigram 2.46.
- `makemore.py` — minimal bigram counting script.

## Run

```bash
pip install torch matplotlib jupyter
python trigram_splits.py
jupyter notebook makemore.ipynb
```
