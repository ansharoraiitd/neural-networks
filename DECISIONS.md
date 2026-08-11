# Decision Log
A record of every technical choice I made and why.

---
## Week 4 — Neural Networks From Scratch
### Why numpy-only before PyTorch
Building forward pass and backprop by hand first means PyTorch's
autograd stops being magic — it's automating exactly the derivative
bookkeeping I'll have done manually. Same philosophy as Week 1's
gradient descent from scratch before using sklearn.