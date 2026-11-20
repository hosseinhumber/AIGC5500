# Client brief and scope sign-off

Submit this file (one page) for approval before you build. After approval it does not change, except that you add
the results table at the end.

## Task and data

- Task: <what is predicted from what>
- Dataset and source: <name, link, licence>
- Split sizes (train / holdout / val / test): <numbers>
- Rare group: <which class or slice, and how many training examples it has>
- Second group: <the measurable input property, how it is computed, the threshold, and how many validation examples fall in it>

## Who the model serves

<Who uses the output, what they do with it, and what a wrong answer on the rare group costs them.>

## Budget (fixed now, before any results)

| Constraint | Limit | Where it is measured |
|---|---|---|
| Time per input | <number> ms (median and 95th percentile) | <named hardware, one thread, batch of 1> |
| Model size | <number> MB | <fp32 parameters, or stated otherwise> |
| Recall on the rare group | at least <number> | validation split |
| <Add one or two more of your own, for example calibration error> | <limit> | <where> |

## Planned components

- Pretrained checkpoint: <name and source>
- Generative model: <type and why it suits this input>
- Decision layer: <the three actions, and a first idea for the reward>

## What counts as done, and who owns what

<One line per piece of the scope with the owner.>

## The two risks most likely to stop us

1. <Risk, and what we will do instead.>
2. <Risk, and what we will do instead.>

## Results against the budget (added at the end)

<Paste the final table from the notebook.>
