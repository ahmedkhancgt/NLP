# Chapter 13: Naive Bayes

## Core Idea
Naive Bayes builds a spam classifier by assuming (naively) that the presence of each vocabulary word is conditionally independent given spam/ham status, which lets Bayes's theorem reduce to multiplying (or, for numerical stability, summing logs of) per-word probability estimates — simple, often effective, and fully implementable from scratch.

## Frameworks Introduced
- **Naive Bayes conditional-independence assumption**: `P(X1,...,Xn | S) = P(X1|S) * ... * P(Xn|S)` — presence/absence of each word is treated as independent given spam status, even though this is "an extreme assumption" (e.g., it ignores that "bitcoin" and "rolex" co-occurring is informative).
  - When to use: text classification (spam filtering) where per-word evidence can be reasonably approximated as independent; works well in practice despite the unrealistic assumption.
- **Additive (Laplace-style) smoothing with pseudocount k**: `P(Xi|S) = (k + spams_containing_wi) / (2k + total_spams)` avoids zero probabilities for words unseen in one class, which would otherwise make the whole message's probability collapse to 0 or 1.
  - When to use: always, when estimating per-token probabilities from finite training counts — prevents a single never-seen word from dominating the prediction.
- **Log-probability summation to avoid underflow**: instead of multiplying many small probabilities (`p1 * ... * pn`), sum their logs and `exp()` the total — numerically safer on floating point.
  - When to use: any model multiplying many probabilities together (recurs in later probabilistic models).
- **Full-vocabulary scoring (presence AND absence)**: for *every* token in the trained vocabulary (not just ones in the message), add `log(p)` if present or `log(1-p)` if absent — absence of expected spam words is itself evidence.

## Key Concepts
- **Spam (S) / ham (¬S)**: the two classes; "ham" is the book's (and field's) conventional term for legitimate, non-spam messages.
- **Token**: a lowercase word extracted via regex (`[a-z0-9']+`) from message text, deduplicated into a `Set[str]` via `tokenize()`.
- **Message NamedTuple**: `Message(text: str, is_spam: bool)` — the labeled training unit.
- **Pseudocount (k)**: smoothing parameter controlling how strongly unseen-word probabilities are pulled toward 0.5; higher k = more smoothing.
- **spam_probability**: `prob_if_spam / (prob_if_spam + prob_if_ham)`, the final Naive Bayes posterior used for classification (threshold at 0.5 by convention).

## Mental Models
- **"Naive" names the independence assumption, not the method's quality**: despite assuming away real word co-occurrence patterns, Naive Bayes "often performs well and has historically been used in actual spam filters" — simplicity and robustness can beat theoretical correctness.
- **Smoothing as insurance against zero-probability catastrophe**: without it, one never-seen word (e.g., "data" appearing in 0/98 spam messages) would force `P(data|S)=0`, making the classifier always predict "not spam" for any message containing that word, however spammy otherwise.
- **Logs turn multiplication into addition for numerical safety**: `log(a*b) = log(a) + log(b)` and `exp(log(x)) = x` — a general pattern for any product of many small probabilities, not specific to Naive Bayes.
- **Feature engineering beyond raw tokens**: the book notes you can inject synthetic tokens (e.g., `contains:number`) into the tokenizer to encode non-word features, showing Naive Bayes's feature space is whatever the tokenizer emits, not just literal words.

## Anti-patterns
- **Estimating P(word|spam) as a raw fraction without smoothing**: a word absent from all spam training examples gets `P=0`, permanently zeroing out spam probability for any message containing it regardless of other evidence — always smooth with a pseudocount.
- **Multiplying many small probabilities directly**: risks floating-point underflow; sum log-probabilities instead.
- **Treating only words present in the message as evidence**: the correct Naive Bayes scoring also uses the probability of *not* seeing each vocabulary word that's absent from the message — skipping this loses information.
- **Training on too little data or too narrow a feature set**: the worked example gets only 77% precision / 65% recall using subject lines alone — looking only at a small slice of available text (and ignoring word frequency, stemming, or minimum-count filtering) caps achievable performance.

## Code Examples
```python
def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

class Message(NamedTuple):
    text: str
    is_spam: bool
```
- **What it demonstrates**: text-to-token preprocessing and the labeled training-example type.

```python
def _probabilities(self, token: str) -> Tuple[float, float]:
    spam = self.token_spam_counts[token]
    ham = self.token_ham_counts[token]
    p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
    p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
    return p_token_spam, p_token_ham

def predict(self, text: str) -> float:
    text_tokens = tokenize(text)
    log_prob_if_spam = log_prob_if_ham = 0.0
    for token in self.tokens:
        prob_if_spam, prob_if_ham = self._probabilities(token)
        if token in text_tokens:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_ham += math.log(prob_if_ham)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_ham += math.log(1.0 - prob_if_ham)
    prob_if_spam, prob_if_ham = math.exp(log_prob_if_spam), math.exp(log_prob_if_ham)
    return prob_if_spam / (prob_if_spam + prob_if_ham)
```
- **What it demonstrates**: smoothed per-token probability estimation and full log-space Naive Bayes scoring over the entire vocabulary.

## Reference Tables
| Component | Formula |
|---|---|
| Smoothed P(token\|spam) | `(spam_count + k) / (total_spams + 2k)` |
| Spam posterior | `P(spam\|msg) = p_if_spam / (p_if_spam + p_if_ham)` |
| Log-space product | `exp(sum(log(p_i) for p_i in probs))` |

| Classification outcome (real dataset) | Count |
|---|---|
| True positive (spam -> "spam") | 84 |
| False positive (ham -> "spam") | 25 |
| False negative (spam -> "ham") | 44 |
| True negative (ham -> "ham") | 703 |
| Precision | 84/(84+25) = 77% |
| Recall | 84/(84+44) = 65% |

## Worked Example
A toy training set of 3 messages (`"spam rules"` spam; `"ham rules"`, `"hello ham"` ham) trains a `NaiveBayesClassifier(k=0.5)`; hand-computing `predict("hello spam")` by multiplying smoothed per-token probabilities for all 4 vocabulary tokens (spam, ham, rules, hello) matches the classifier's own output of ~0.83 probability of spam. On the real SpamAssassin corpus (subject lines only, 75/25 train/test split, seed 0), the trained model achieves 77% precision and 65% recall; inspecting `p_spam_given_token` for each vocabulary word surfaces "spammiest" words (sale, mortgage, money, rates) and "hammiest" words (spambayes, users, apt, perl), confirming the model is capturing sensible signal.

## Key Takeaways
1. Naive Bayes reduces a joint word-probability estimate to a product (sum of logs) of independent per-word estimates — simple, fast, and often effective for text classification despite the unrealistic independence assumption.
2. Always apply pseudocount smoothing when estimating probabilities from finite counts, or unseen words will create zero-probability catastrophes.
3. Sum log-probabilities instead of multiplying raw probabilities to avoid floating-point underflow.
4. Score using the *entire* vocabulary (absence as well as presence of each token), not just words that appear in the message.
5. Model performance can often be improved via richer features (full message body, minimum token count filtering, stemming, synthetic tokens) rather than more sophisticated algorithms.

## Connects To
- **Ch6 (Probability)** and **Ch6's Bayes's theorem**: the entire chapter is a direct, concrete application of Bayes's theorem from Ch6.
- **Ch11 (Machine Learning)**: reuses `split_data` and the precision/recall/confusion-matrix evaluation framework.
- **Ch21 (NLP)**: tokenization here is the simplest form of the text-processing techniques expanded later.
- **Porter stemmer / scikit-learn's BernoulliNB**: external tools the book points to for stemming and a production-grade equivalent implementation.
