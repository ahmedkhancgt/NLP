# Chapter 23: Recommender Systems

## Core Idea
The chapter builds four increasingly sophisticated recommendation strategies — manual curation, popularity-based, user-based collaborative filtering, item-based collaborative filtering — then introduces matrix factorization (learning latent user/item embeddings via gradient descent) and applies it to the real MovieLens 100k dataset to predict movie ratings.

## Frameworks Introduced
- **Popularity-based recommendation**: recommend the globally most common items the user doesn't already have.
  - When to use: cold-start (no personalization data) or as a trivial baseline — "lots of people like X" isn't a compelling personalized pitch.
- **User-based collaborative filtering**: represent each user as a 0/1 interest vector; find similar users via `cosine_similarity`; recommend items weighted by the sum of similarities of other users who have them.
  - When to use: personalizing recommendations based on who you resemble — breaks down in high dimensions (curse of dimensionality, Ch12) since "most similar user" may still be far away when there are many possible items.
- **Item-based collaborative filtering**: transpose the user-interest matrix so rows are items; compute item-item cosine similarity; recommend items similar to the user's *existing* interests, weighted by similarity.
  - When to use: an alternative to user-based filtering that often scales/generalizes better since item-item similarity is typically more stable than user-user similarity.
- **Matrix factorization (latent-factor model)**: assume every user and every item has a latent "type" vector (`dim`-dimensional); predicted rating = `dot(user_vector, item_vector)`; learn both sets of vectors jointly via gradient descent to minimize squared prediction error on observed ratings.
  - When to use: predicting numeric ratings (not just binary interest) at scale, especially when most user-item pairs are unobserved (sparse rating matrices like MovieLens).

## Key Concepts
- **Interest vector**: a 0/1 vector over a sorted `unique_interests` vocabulary, 1 if the user listed that interest.
- **Cosine similarity for binary vectors**: reused from Ch21; identical interest sets give similarity 1, disjoint sets give similarity 0.
- **Latent type / embedding**: an unobserved, learned vector representation capturing some notion of "kind of user" or "kind of item" whose dot product predicts affinity — explicitly likened to Ch21's word embeddings.
- **Train/validation/test split for recommenders**: 70/15/15 split of ratings used to fit embeddings, tune hyperparameters (embedding dimension, learning rate schedule), and finally evaluate.
- **Baseline model**: "always predict the average rating," whose mean squared error (here ~1.265) is the bar any real model must beat.
- **Decreasing learning rate schedule**: multiply the learning rate by 0.9 each epoch, empirically found to improve matrix-factorization training.

## Mental Models
- **Four strategies on a spectrum of personalization and cost**: manual curation (fully bespoke, doesn't scale) -> popularity (zero personalization, trivial) -> user-based CF (personalized via similar people) -> item-based CF (personalized via similar items, often more robust) -> matrix factorization (learns latent structure directly from ratings, scales to sparse large datasets).
- **User-based CF's curse-of-dimensionality failure mode**: with thousands of possible items (like Amazon's full catalog), "my most similar shopper" may not resemble me at all — the same high-dimensional-sparsity problem from Ch12's k-NN discussion.
- **Matrix factorization as "compressing" the preference matrix**: rather than directly modeling `num_users x num_items` entries, factor that matrix into a `num_users x dim` and `dim x num_items` product — learning dense, lower-dimensional representations that generalize to unobserved pairs.
- **Interpreting learned embeddings via PCA**: the first principal component of learned movie vectors ends up correlating strongly with average rating ("how good is this movie?") even though no label said so directly — an emergent, interpretable structure from an unsupervised-feeling training signal.

## Anti-patterns
- **Relying solely on user-based collaborative filtering in high-dimensional item spaces**: as item count grows very large, "most similar user" similarity can become meaningless (same curse of dimensionality as k-NN) and recommendations degrade.
- **Treating recommendation weights as intrinsically meaningful numbers**: "the weights aren't intrinsically meaningful; we just use them for ordering" — don't over-interpret the magnitude of a similarity-weighted suggestion score.
- **Skipping a baseline model**: always compute a trivial baseline (e.g., "predict the average") before trusting that a more complex model (matrix factorization) is actually adding value.
- **Overfitting high-dimensional embeddings without regularization**: the author notes the matrix factorization model is "pretty apt to overfit," with the best empirical result coming from a small `EMBEDDING_DIM=2` rather than a larger dimension, and regularization (Ch15 style weight-shrinkage) not improving things further in this case.
- **Treating movie/user IDs as integers when they're not consecutive**: wastes dimensions/creates sparsity artifacts — treat them as opaque string keys instead.

## Code Examples
```python
def user_based_suggestions(user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
    suggestions = sorted(suggestions.items(), key=lambda p: p[-1], reverse=True)
    return suggestions if include_current_interests else \
        [(s, w) for s, w in suggestions if s not in users_interests[user_id]]
```
- **What it demonstrates**: aggregating weighted votes from similar users to produce ranked recommendations.

```python
def loop(dataset: List[Rating], learning_rate: float = None) -> None:
    for rating in dataset:
        movie_vector, user_vector = movie_vectors[rating.movie_id], user_vectors[rating.user_id]
        predicted = dot(user_vector, movie_vector)
        error = predicted - rating.rating
        if learning_rate is not None:
            user_gradient = [error * m_j for m_j in movie_vector]
            movie_gradient = [error * u_j for u_j in user_vector]
            for j in range(EMBEDDING_DIM):
                user_vector[j] -= learning_rate * user_gradient[j]
                movie_vector[j] -= learning_rate * movie_gradient[j]
```
- **What it demonstrates**: matrix factorization's gradient-descent training loop, jointly updating a user embedding and a movie embedding from each observed rating's prediction error.

## Reference Tables
| Strategy | Personalization | Scales to many items? | Needs numeric ratings? |
|---|---|---|---|
| Manual curation | fully bespoke | no | no |
| Popularity | none | yes | no |
| User-based CF | yes (via similar users) | poorly (curse of dimensionality) | no (binary works) |
| Item-based CF | yes (via similar items) | better than user-based | no (binary works) |
| Matrix factorization | yes (learned latent factors) | yes (handles sparse large data) | yes (numeric ratings) |

## Worked Example
On the 15-user `users_interests` dataset, user-based CF recommends `MapReduce`, `MongoDB`, `Postgres`, `NoSQL` to user 0 (interested in Big Data/Hadoop-adjacent topics), based on weighted votes from similar users (most similar: user 9 at 0.567 cosine similarity). Item-based CF, finding interests similar to "Big Data" itself (Hadoop 0.816, Java 0.667, MapReduce 0.577...), produces a similar but distinctly-weighted recommendation list for user 0. On the real MovieLens 100k dataset (943 users, 1682 movies), a baseline "predict the average rating" gets MSE ≈1.265; a 2-dimensional matrix factorization model trained via gradient descent (20 epochs, decaying learning rate from 0.05) achieves test MSE ≈0.89. PCA on the learned 2D movie embeddings shows the first principal component strongly correlates with average movie rating (top 25 movies by that component are highly rated; bottom 25 are poorly rated or unrated), while the second component's meaning remains unclear.

## Key Takeaways
1. Recommendation strategies range from zero-personalization (popularity) to fully learned latent structure (matrix factorization) — pick based on available data and item-space scale.
2. User-based collaborative filtering suffers the curse of dimensionality as the item catalog grows; item-based filtering is often more robust at scale.
3. Matrix factorization learns dense latent user/item embeddings via gradient descent, generalizing to unobserved user-item pairs — the standard approach for sparse rating data.
4. Always benchmark against a trivial baseline (predict the mean) before trusting a more sophisticated model's improvement.
5. Learned embeddings can be interpreted post hoc via PCA, often revealing one dominant, intuitively meaningful axis (e.g., "overall quality") even without being told to find one.

## Connects To
- **Ch21 (NLP)**: cosine similarity is reused verbatim; the latent-type/embedding idea in matrix factorization is explicitly likened to word embeddings.
- **Ch12 (k-Nearest Neighbors)**: user-based CF's degradation in high dimensions is the same curse-of-dimensionality phenomenon discussed there.
- **Ch8 (Gradient Descent)**: matrix factorization's training loop is a direct application of the book's general gradient-descent machinery.
- **Ch10 (PCA)**: used to interpret/visualize the learned movie embeddings.
- **Ch15's Regularization**: considered (but found unhelpful here) as a way to combat matrix-factorization overfitting.
- **Surprise (Python library) / the Netflix Prize**: external resources for production recommender systems and historical context on the field's competitive benchmark.
