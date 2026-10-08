# Final project: An alignment experiment with a measurable gold reward

## Overview

The final project aligns a small language model and measures whether the alignment worked. Students write a gold reward they do not train on, generate preference data with a stated annotator model, align the model with two methods on matched data and compute, push at least one method until the gold reward stops improving and evaluate the result with intervals, a length control and an independent judge. The project is done alone or in pairs.

## Learning outcomes assessed

The project assesses the course learning outcomes on formulating generation as a decision process, reward modelling from preferences, policy optimisation under a KL penalty, direct preference optimisation and the evaluation of aligned models.

## Options

### Option 1: A new corpus and a new gold reward

Replace the corpus of Day 4 with sentences of a domain of your choice and write a gold reward that prefers a style you can state precisely.

### Option 2: RLHF against DPO under annotator noise

Repeat the comparison of Day 5 at three levels of annotator noise and report how the gap between the methods changes.

### Option 3: Hunting the reward hack

Train reward models of different sizes and amounts of data and measure where the over-optimisation peak moves, with a diagnosis of what the policy learns to exploit.

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
