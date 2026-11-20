# <Project title>

<One sentence: the task, and the question the client asked.>

## The question and the recommendation

<Two or three sentences: what the client needs, what you recommend (a named model, or "do not ship"), and the one reason that most decides it. Fill this in last.>

## Data

<Dataset, source, how to obtain it (a command or a link, because the data is not committed), the four split sizes, the rare group, the second group.>

## How to run

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests
jupyter nbconvert --to notebook --execute notebooks/final_project.ipynb --output executed.ipynb
```

<Add anything else needed to reproduce the results, and how long a full run takes on which hardware.>

## Hardware and versions

<CPU, RAM, operating system, Python and torch versions, whether a GPU was used for training, and what else was running during timing.>

## Results summary

<A short table of the candidates against the budget from BRIEF.md. The full tables are in results/.>

## Video presentation

<Link to the video on Humber OneDrive. Presenter: name. Length: mm:ss.>

## Contributions

| Piece | Owner | Account |
|---|---|---|
| 1. Data, baseline and training foundations | <name> | <GitHub username> |
| 2. Adaptation of a pretrained model | <name> | <GitHub username> |
| 3. Generated data | <name> | <GitHub username> |
| 4. Decision layer | <name> | <GitHub username> |
| 5. Compression and cost | <name> | <GitHub username> |
| 6. Trust and the decision | <name> | <GitHub username> |

## References and things we did not write

<Every tutorial, repository, paper, checkpoint and AI assistant whose code or text you used, with what you used it for.>
