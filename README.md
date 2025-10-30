# Spam_Ham_Positive_Negative-Classify

## Description
A small collection of Python scripts and datasets for performing:
- SMS/email spam vs. ham classification (TF–IDF + LinearSVC).
- Movie review sentiment classification (IMDB dataset, TF–IDF + LinearSVC).
The repository includes training and prediction scripts, example datasets, and a simple email analysis helper.

## Project files
- [email_analyzer.py](email_analyzer.py)
- [spam-ham_train.py](spam-ham_train.py)
- [spam-ham_predict.py](spam-ham_predict.py)
- [sentiment_train.py](sentiment_train.py)
- [sentiment_predict.py](sentiment_predict.py)
- [email.csv](email.csv)
- [IMDB Dataset.csv](IMDB Dataset.csv)
- [some_email.txt](some_email.txt)
- [requirements.txt](requirements.txt)
- [README.md](README.md)

## Setup instructions
1. Create and activate a virtual environment:
   - Windows (PowerShell)
     ```
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - macOS / Linux
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (See [requirements.txt](requirements.txt))

3. Confirm datasets are present:
   - [email.csv](email.csv) — SMS/email labels (spam/ham)
   - [IMDB Dataset.csv](IMDB Dataset.csv) — movie reviews with sentiments

## Usage
- Train a spam/ham model:
  ```
  python spam-ham_train.py
  ```
  Output: model and vectorizer files (check script for filenames).

- Predict with a trained spam/ham model:
  ```
  python spam-ham_predict.py --model path/to/model.joblib --vectorizer path/to/vect.joblib --input "Free prize! Call now"
  ```

- Train a sentiment model:
  ```
  python sentiment_train.py
  ```

- Predict sentiment:
  ```
  python sentiment_predict.py --model path/to/sentiment_model.joblib --vectorizer path/to/sent_vect.joblib --text "This movie was excellent!"
  ```

- Quick email analysis:
  ```
  python email_analyzer.py some_email.txt
  ```

Open the training scripts to see configurable options:
- [`spam-ham_train.py`](spam-ham_train.py)
- [`sentiment_train.py`](sentiment_train.py)

## Contributor guidelines
- Fork the repository and create feature branches.
- Keep commits small and focused; use clear commit messages.
- Follow PEP 8 for Python code.
- Add or update minimal examples or usage instructions when changing behavior.
- Create a PR with a description of changes and link to related issue (if any).

No automated tests are included; add tests under a `tests/` folder and include instructions in this README if added.

## License
This project is provided under the MIT License. Add a `LICENSE` file to the repository with the MIT text if you want to apply it officially.