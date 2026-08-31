# Data

`LSP_r1300_p40_small.h5` (390 MB) is not committed here - GitHub blocks files over 100 MB, and Git LFS's free tier (1 GB storage / 1 GB bandwidth per month, account-wide) would be mostly consumed by a single file that size, with only 2-3 downloads a month before the whole account hits its bandwidth limit. Not worth that risk for a portfolio repo meant to be cloned by many people.

Download it yourself instead:

```
pip install kaggle
kaggle datasets download samueljbaker/enginebench-lsp-small --unzip -p .
```

Requires a free Kaggle account and API token (kaggle.com → Settings → API → Create New Token). See the project README's "How to reproduce" section for the full path.
