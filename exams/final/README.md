# Final project: An alignment experiment with a measurable gold reward

## Overview

The final project aligns a small language model and measures whether the alignment worked. Students write a gold reward they do not train on, generate preference data with a stated annotator model, align the model with two methods on matched data and compute, push at least one method until the gold reward stops improving and evaluate the result with intervals, a length control and an independent judge. The project is done alone or in pairs.

## Learning outcomes assessed

The project assesses the course learning outcomes on formulating generation as a decision process, reward modelling from preferences, policy optimisation under a KL penalty, direct preference optimisation and the evaluation of aligned models.

## Options

Each option comes with a short first step in Python. The examples run in Colab as they are and stop where the project begins.

### Option 1: A new corpus and a new gold reward

Replace the corpus of Day 4 with sentences of a domain of your choice and write a gold reward that prefers a style you can state precisely.

**A first step.** The code below writes a small corpus of weather reports, a gold reward for a stated style and a reference model that continues every word with a word that followed it in the corpus.

```python
import random
from collections import defaultdict

corpus = ["rain is likely in the north <eos>",
          "sunny and warm with 28 degrees <eos>",
          "cloudy with light wind in the north <eos>",
          "sunny in the south with 31 degrees <eos>",
          "light rain and 19 degrees in the south <eos>"]

def gold_reward(sentence):
    """The style to prefer: short reports that state a temperature."""
    words = [w for w in sentence.split() if w != "<eos>"]
    return 1.0 * any(w.isdigit() for w in words) - 0.1 * len(words)

# A reference model from the corpus: the words that follow each word
follow = defaultdict(list)
for s in corpus:
    words = ["<bos>"] + s.split()
    for a, b in zip(words, words[1:]):
        follow[a].append(b)

def sample(rng, max_words=12):
    word, out = "<bos>", []
    while word != "<eos>" and len(out) < max_words:
        word = rng.choice(follow[word])
        out.append(word)
    return " ".join(out)

rng = random.Random(0)
for _ in range(6):
    s = sample(rng)
    print(f"{gold_reward(s):+.2f}  {s}")
```

### Option 2: RLHF against DPO under annotator noise

Repeat the comparison of Day 5 at three levels of annotator noise and report how the gap between the methods changes.

**A first step.** The code below simulates an annotator who picks the worse of two answers with a given probability, and measures how many of the resulting pairs still agree with the gold reward.

```python
import numpy as np

rng = np.random.default_rng(0)
gold = rng.normal(size=200)          # gold rewards of 200 candidate answers

def annotate(n_pairs, noise):
    """Pairs (winner, loser). With the probability `noise` the annotator picks the worse answer."""
    i, j = rng.integers(0, len(gold), (2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    pick_i = (gold[i] > gold[j]) ^ (rng.random(len(i)) < noise)
    return np.where(pick_i, i, j), np.where(pick_i, j, i)

for noise in (0.0, 0.1, 0.3):
    winner, loser = annotate(1000, noise)
    agree = np.mean(gold[winner] > gold[loser])
    print(f"noise {noise:.1f}: {len(winner)} pairs, {agree:.3f} of them agree with the gold reward")
```

### Option 3: Hunting the reward hack

Train reward models of different sizes and amounts of data and measure where the over-optimisation peak moves, with a diagnosis of what the policy learns to exploit.

**A first step.** The code below picks the best of n answers by a proxy reward that also rewards flattery. The gold reward of the chosen answer first rises with n and then falls.

```python
import numpy as np

rng = np.random.default_rng(1)

def sample_answers(n):
    quality = rng.normal(size=n)
    flattery = np.abs(rng.normal(size=n))      # how much an answer flatters the reader
    gold = quality - 0.5 * flattery ** 2       # people value quality and dislike heavy flattery
    proxy = quality + 1.0 * flattery           # a reward model that also rewards flattery
    return gold, proxy

for n in (1, 2, 4, 8, 16, 64, 256):
    picked = []
    for _ in range(1000):
        gold, proxy = sample_answers(n)
        picked.append(gold[np.argmax(proxy)])  # best of n against the proxy reward
    print(f"best of {n:3d}: mean gold reward {np.mean(picked):+.3f}")
```

## Proposal

Before starting, send a proposal of about 150 words that names the option, the dataset and the question the project answers. The proposal is optional during self-study and expected during an active delivery.

## Deliverables

The submission consists of one Colab notebook that runs from top to bottom without errors, and a technical report of 2000 to 3000 words that follows [the report template](../REPORT_TEMPLATE.md). Figures in the report must be produced by the notebook.

## Evaluation

| Criterion | Weight |
|---|---|
| The decision process is stated correctly and the code matches it | 15 % |
| The gold reward is defensible and genuinely held out of training | 20 % |
| Both methods are compared on matched data and matched compute | 20 % |
| The over-optimisation study locates a peak and diagnoses the mechanism | 25 % |
| Evaluation has intervals, a length control and an independent judge | 20 % |

## Submission

During an active delivery of the course, the notebook and the report are sent within one week after Day 5 to utkukose@sdu.edu.tr or utkukose@gmail.com, with the subject line "VTR UGE 21 Reinforcement Learning and Language Model Alignment final project".
