[![Python package](https://github.com/Eve-ning/snn_voice/actions/workflows/python-package.yml/badge.svg)](https://github.com/Eve-ning/snn_voice/actions/workflows/python-package.yml)

# Speech Command predictions through SNN

SNN-CNN Hybrid Model for Voice Recognition

For our NTU FYP, we explore the efficacy of using SNN for dimensionality reduction (thus parameter reduction) as a preprocessing step.
This heavily reduces required hardware to realise the system, however, what are the tradeoffs to be made?

## Modular Experiment Structure

We segment our experiments modularly: That means, for each purpose, 
we can substitute parts that fulfil that purpose.

| Section    | Purpose           | Part A           | Part B      |
|------------|-------------------|------------------|-------------|
| Input      | Ingestion         | Speech Commands  | TIDIGTS     |
|            | Preprocessing     | Resampling       | Spectrogram |
| Model      | Structure         | M5               | Piczak      |
|            | Optimization      | Adam             | SGD         |
|            | Scheduling        | Cosine Annealing | Step LR     |
| Evaluation | Metric Monitoring | Top-K (Accuracy) |             |

Take for example, we can run a Piczak Experiment through
- Speech Commands Dataset
- Resampling to 4000Hz & transforming with Mel Spectrogram
- Using the Piczak Model structure
- Using Adam to optimize
- With a Cosine Annealing Learning Rate Scheduler
- Finally, evaluating with Top-K Accuracy

> You can see how it works in our unit tests!

## Tying back to SNNs

Here, we're interested in substituting some CNN layers with SNNs. 
We can simply change **Model Structure** and retain the rest, 
this heavily reduces redundancy.
