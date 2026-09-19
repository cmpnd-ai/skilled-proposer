# Attribution

The file `prompts.jsonl` is a stratified sample of 1,200 rows from the
AllenAI dataset `allenai/RLVR-IFeval` (part of the Tulu 3 release), which
pairs prompts from the Tulu 2 SFT mixture with verifiable constraints from
IFEval. The checkers in `verifiers.py` follow the pass and fail logic of the
`if_functions.py` module in AllenAI's open-instruct repository,
https://github.com/allenai/open-instruct, under the Apache License 2.0.
The language detection constraint is omitted because the sample has no
rows of that type.
