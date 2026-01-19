+++
title = "Trajectory"
date = 2026-01-19
draft = false
description = "You won each decision and lost the trajectory."
tags = []
+++

People optimize for local objectives and lose sight of global ones. The shape is always the same: you win at the thing
directly in front of you, and in doing so, you lose something larger.

Most tasks are very local, and in ML this has become extreme. We train on language and evaluate on language. It's the
same domain, so it must be fair. But "all human knowledge" turned out to be larger than anticipated, and code
performance doesn't improve with math or translation.

Consider ImageNet, which has a proper train/test split. While foundational techniques like BatchNorm, residual
connections and dropout transferred, more specific tweaks did not. For example, depth-wise separable convolutions
aren't a general backbone, as they don't work in GANs. People say this is because GANs are a new domain with new
requirements, but that's too simplistic. The more aggressive formulation: it doesn't transfer.

If I invent AdamR, tuned for ResNet-18 on ImageNet, which doesn't transfer to CIFAR or language models, is it truly
useful? It's not a general algorithm. That's why Adam is still king. Not because Adam is optimal for any particular
task, but because it doesn't require extensive tuning to work with transformers, ResNets, CIFAR, or Common Crawl.

The pattern is obvious retrospectively: CNNs "just didn't work on some problems." LSTMs failed to generalize to vision.
Both showed early signs they wouldn't generalize. Those failures were the signal they wouldn't keep working on new
domains. They went down as hacks that help early convergence, not as methods that stood the test of time.

If your method doesn't work across existing domains, it won't work for the next one either.

Taking a new job optimizes for local reward. Salary, status, interesting problems. But each job reshapes you. You adopt
its values, its rhythms, its definitions of success. A few years in, you realize you've been optimized by the job more
than you've optimized your career. 

You won each decision and lost the trajectory.