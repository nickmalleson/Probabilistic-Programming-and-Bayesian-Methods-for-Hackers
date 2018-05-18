find Prologue Chapter* -name "*.ipynb" -execdir jupyter nbconvert --to pdf --template article --output-dir /Users/nick/Dropbox/research/reading/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/ {} \;

# merge all files:
pdfjoin Prologue.pdf Ch*.pdf DontOverfit.pdf MachineLearning.pdf
