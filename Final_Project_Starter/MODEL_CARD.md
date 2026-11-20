# Model card: <model name>

Write this for the model you recommend, or for the best candidate if you recommend that none is shipped.

## Intended use

<Who uses it, for what, on what hardware, and what a person does with a prediction.>

## Not intended for

<At least one situation in which this model should not be used, and why.>

## Data

<Source, size, the four splits, how the rare group and second group are defined, and what the data does not cover.>

## Model

<Architecture, how it was trained or adapted, parameter count, size on disk.>

## Performance

<Headline metric on the validation and (once) test split, with every seed or the spread. Then the same metric for the rare group and the second group.>

## Cost

<Time per input (median and 95th percentile, hardware, threads, batch), peak memory.>

## Calibration, robustness and explanation

<ECE before and after temperature scaling, accuracy under each corruption, and what the sanity check of the explanation showed.>

## Known problems and what we did about them

<Gaps between groups, the mitigation tried, what it fixed and what it cost.>

## What we do not know

<The things this evidence cannot establish.>
