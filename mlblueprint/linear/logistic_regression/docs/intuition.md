# Logistic Regression — Intuition

## The Big Idea

Imagine you're a bank deciding whether to approve a loan application. You have the applicant's income and credit score, and you know from past data whether similar applicants paid back or defaulted.

| Income ($k) | Credit Score | Repaid? |
|---|---|---|
| 80 | 750 | Yes |
| 35 | 580 | No |
| 120 | 800 | Yes |
| 25 | 520 | No |

Linear regression would try to draw a straight line through this, but that doesn't make sense — the answer isn't a number, it's **yes or no**. What you really want is a **probability**: "Given this income and credit score, what's the chance they'll repay?"

**That's what Logistic Regression does.** It takes a weighted sum of your features plus a bias and squashes it through an S-shaped curve (the sigmoid) to give you a probability between 0 and 1. The full maths lives in [derivation](derivation.md).

## What is it actually doing?

Start with the same linear score as linear regression: each feature gets a weight, plus one bias term. Large positive scores mean "very likely yes", large negative scores mean "very likely no", and zero means "fifty-fifty".

The sigmoid turns that score into a probability. Near zero the curve is steep, so small changes matter a lot. Far from zero it flattens out, so pushing an already-confident score further barely changes the probability.

To learn, we pick the weights that make the observed labels most likely. The loss for that (log-loss) punishes confident wrong answers far more than cautious ones: predicting 0.99 when the truth is 0 hurts much more than predicting 0.6. Training is gradient descent on that loss, and the gradient turns out to be just the prediction error (probability minus label) times the features — the same shape as linear regression, with probabilities in place of raw predictions.

## When would you use it?

- Predicting whether an email is spam (yes/no)
- Medical diagnosis: disease present or absent based on test results
- Credit approval: will this person default?
- Any binary classification problem where you want probabilities, not just hard labels

## When would you NOT use it?

- When a straight line can't separate the classes and you need a curved boundary (trees or neural nets handle this better)
- When you have more than two classes without modification (need multinomial/softmax)
- When you need complex feature interactions without engineering them first

## The key takeaway

Logistic regression is linear regression that went to probability school. It keeps the simplicity and interpretability of a linear model but outputs calibrated probabilities. The only real trick is keeping the log-loss numerically stable when probabilities get close to 0 or 1.
