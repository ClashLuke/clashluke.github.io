+++
title = "The Box"
date = 2026-01-01
draft = false
description = "What happens when you revisit the assumptions everyone forgot were assumptions?"
tags = []
+++

What happens when you revisit the assumptions everyone forgot were assumptions?

Language models predict the next token because token-level loss is tractable, not because token-level prediction captures what we want models to learn. Transformers use fixed context windows because attention is quadratic in sequence length, not because 128k tokens is enough context. We train on dense rewards because sparse rewards are hard to optimize, not because the problems we care about have dense rewards. These choices accumulate into a box that everyone works inside, and after enough time, people forget the box is there at all.

I spent almost three years at [Keen Technologies](https://keenagi.com/) working on AGI. I left because I kept being drawn to a different question. The question of what happens when you revisit these assumptions.

What happens when agents share knowledge instead of each learning from scratch? What happens when training objectives match test objectives even for sparse rewards? What happens when you remove the assumptions that force local optimization and find structures that permit global coherence?

What happens when you step outside the box?
