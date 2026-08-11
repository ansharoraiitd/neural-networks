# Week 4 Summary: Neural Networks From Scratch

## The throughline from Week 1

A single neuron (Monday) is exactly Week 1 Saturday's logistic
regression — same equation, same sigmoid, same gradient descent.
Everything this week is "what happens when you stack many of those
together and need the chain rule to train them."

## What connects to what

- **Backprop (Tue) = gradient descent (Week 1 Fri/Sat), extended
  via the chain rule** to handle multiple layers instead of one
  equation.
- **Vanishing gradients (Mon/Tue) are the SAME bias-variance-style
  tradeoff** as every complexity knob since Week 2 — depth helps
  representational power but creates its own failure mode
  (gradients shrinking to nothing), same shape of tradeoff, new
  mechanism.
- **Softmax + cross-entropy (Wed) generalizes sigmoid + log-loss**
  (Week 1 Sat) from 2 classes to N — same clean combined gradient
  form, same reason they're always paired.
- **Adam (Wed) is momentum + Week 1's gradient descent, made
  adaptive per-parameter** — not a different algorithm, an
  extension of the same update-the-weights-downhill idea.
- **CNNs (Thu) solve the SAME curse-of-dimensionality problem**
  KNN hit in Week 2 Monday — too many parameters, not enough
  structure — via parameter sharing instead of dimensionality
  reduction.
- **Confusion matrices (today) are Week 1 Thursday's precision/
  recall, run once per class** — not new math, the same metrics
  applied N times with macro/weighted averaging added.

## Biggest "aha" this week
[fill this in yourself, in your own words, before moving on —
this line is worth more for interview prep than any code]

## What I'd still like to go deeper on
[same — one honest, specific gap, not "everything"]