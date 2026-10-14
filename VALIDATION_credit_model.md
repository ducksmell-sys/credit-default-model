# Model validation report: consumer credit default (PD) model

Scope: the logistic regression PD model in `credit_models.py`, its data and its use. Structure follows
the three areas of SR 11-7: conceptual soundness, ongoing monitoring (stability) and outcomes analysis.

**Independence note.** The same person developed and validated this model. A bank would require a
separate validation team. The mitigations are acceptance criteria fixed in `validate_credit_model.py`
before the full script was written (after a smoke run that had shown the headline AUCs; they were not
adjusted to it) and reporting every failed criterion.

## 1. Model overview and intended use

| Item | Description |
|---|---|
| Purpose | Estimate the probability that a credit card customer defaults in the next month, to rank customers by risk and support credit decisions |
| Candidate | Logistic regression, 22 features, standardized inputs, L2 regularization (C = 1) |
| Challenger | Gradient boosting (histogram-based, depth 4, early stopping) on the same features |
| Benchmark | Logistic regression on last month's repayment status only |
| Excluded inputs | `SEX`, `MARRIAGE`, `AGE` (protected attributes under common fair-lending practice) |
| Assumed use | Ranking and approving or declining at a risk cut-off (the validation uses "approve the 80% lowest-PD customers") |

## 2. Data

UCI "Default of credit card clients" (Taiwan, 2005): 30,000 customers, 22.12% default rate, split
18,000 / 6,000 / 6,000 into train, validation and test (stratified, seed 42). The data has no dates.

| Check | Result |
|---|---|
| Missing cells | 0 |
| Duplicate rows | 56 |
| Last-month repayment codes -2 and 0 (not defined in the dataset documentation) | 17,496 rows |
| Education codes 0, 5, 6 (undocumented) | 345 rows |
| Marriage code 0 (undocumented) | 54 rows |
| Negative bill amount (last month) | 590 rows |
| Bill above credit limit (last month) | 2,115 rows |

## 3. Conceptual soundness

**Sign test.** Sixteen features have a directional expectation from credit logic (more delay or
utilization raises default, more payment or a higher limit lowers it). Seven coefficients contradict it:
`DELAY_2` to `DELAY_6` (negative), `UTIL_AVG` (negative) and `PAY_RATIO_1` (positive).

**Collinearity.** Variance inflation factors above 10: `N_LATE_MONTHS` (18.2), `EDU_2` (15.8) and
`EDU_1` (14.6). The late-month count overlaps with the individual delay months, which explains the
negative signs: the model uses the count and then corrects with the months.

**Follow-up.** Dropping `DELAY_2` to `DELAY_6`, `UTIL_AVG`, `PAY_RATIO_1` and the education dummies leaves
12 features. Result: zero sign mismatches, no VIF above 10, validation AUC 0.7627 (candidate 0.7625), test
AUC 0.7758 (candidate 0.7753), test KS 0.4222 (candidate 0.4183). The comparison was made on validation
AUC and sign logic; the test figures confirm it.

**Education coefficients.** Measured against an "other" reference group of 345 records, odds ratios are
about 1.5 to 1.9 per standard deviation, and the bootstrap standard deviation of each coefficient
(0.09 to 0.13) is the widest among the ten largest drivers. Removing the education group changes test AUC by only -0.0008.

## 4. Outcomes analysis

**Discrimination (AUC / KS).**

| Model | Train | Validation | Test |
|---|---|---|---|
| Logistic (candidate) | 0.780 / 0.427 | 0.763 / 0.395 | 0.775 / 0.418 |
| Gradient boosting (challenger) | 0.807 / 0.459 | 0.774 / 0.415 | 0.791 / 0.443 |
| Benchmark (last month's status only) | 0.717 / 0.376 | 0.692 / 0.349 | 0.713 / 0.387 |

The candidate's test AUC has a 95% bootstrap interval of 0.761 to 0.790. The overfitting gap (train minus
test AUC) is 0.004 for the candidate and 0.016 for the challenger.

**Challenger and benchmark (test, paired bootstrap).** Challenger minus candidate: +0.0154 AUC (95%
interval +0.0098 to +0.0218). Candidate minus benchmark: +0.0619 (interval +0.0491 to +0.0742).

**Calibration (test).** Candidate mean PD 0.2210 against an observed 0.2212; calibration slope 0.987,
intercept -0.013, Brier 0.1351; Hosmer-Lemeshow 11.23, p = 0.19. Challenger: slope 1.043, Brier 0.1316,
p = 0.34. By decile the candidate is close throughout; the largest gap is decile 3 (predicted 9.8%,
observed 7.3%, n = 600, about 2.0 standard errors).

**Segments (test, candidate).**

| Segment | n | Observed | AUC |
|---|---|---|---|
| Limit: lowest quartile | 1,523 | 30.9% | 0.741 |
| Limit: highest quartile | 1,442 | 14.6% | 0.777 |
| Education: graduate school | 2,141 | 19.2% | 0.788 |
| Education: high school | 989 | 26.5% | 0.745 |
| Last month not late | 4,638 | 13.6% | **0.658** |
| Last month late | 1,362 | 51.1% | 0.726 |
| Utilization below 30% | 3,007 | 18.8% | 0.738 |
| Utilization 30% to 80% | 1,447 | 24.2% | 0.855 |
| Utilization 80% or more | 1,546 | 26.7% | 0.732 |

All segments clear the 0.65 threshold, the "not late last month" segment only narrowly.

## 5. Stability and sensitivity

- **PSI (train vs test):** score 0.00103; largest feature PSI 0.00246 (`DELAY_6`). The split is random, so
  this shows the monitoring calculation works and says nothing about drift over time.
- **Coefficient stability:** in 200 bootstrap refits the sign of each of the five largest coefficients is
  unchanged in 100% of refits. `PAY1_LATE1` is the one feature whose sign is unstable (small and
  collinear).
- **Feature-group ablation (test AUC, candidate 0.7753):** without recent status 0.7615; without delay
  history 0.7624; without limit and balance size 0.7701; without utilization 0.7717; without payment
  behaviour 0.7716; without education 0.7745. The model relies mainly on repayment history.
- **Input noise:** 10% multiplicative noise on limit, bills and payments (20 draws) lowers test AUC by
  0.0002.

## 6. Fairness screen

Protected attributes are not inputs. Cut-off: approve if PD is at most 0.3015 (the 80% quantile on
validation), giving an overall test approval rate of 79.0%. Adverse impact ratio (AIR) is the group
approval rate divided by the highest group approval rate within the attribute.

| Group | n | Approval | AIR | Observed | Mean PD | Calibration z |
|---|---|---|---|---|---|---|
| Male | 2,372 | 77.7% | 0.97 | 24.1% | 22.5% | +1.9 |
| Female | 3,628 | 80.0% | 1.00 | 20.8% | 21.8% | -1.5 |
| Under 30 | 1,926 | 76.0% | 0.93 | 21.9% | 23.9% | -2.1 |
| 30 to 44 | 2,952 | 81.7% | 1.00 | 20.8% | 20.5% | +0.4 |
| 45 and over | 1,122 | 77.4% | 0.95 | 26.0% | 23.2% | +2.2 |
| Married | 2,731 | 78.8% | 0.99 | 23.9% | 22.1% | +2.2 |
| Single | 3,196 | 79.3% | 1.00 | 20.7% | 22.1% | -1.9 |

(The "other or unknown" marriage group has 73 customers and is omitted from the table; AIR 1.00.)

All AIR values are at least 0.93, above the 0.80 screening heuristic. Three group results have a
calibration gap of two or more standard errors (under 30, 45 and over, married). With eight results about
0.4 would be expected by chance if they were independent, though groups within an attribute are not.

## 7. Summary against acceptance criteria

Ten of twelve criteria pass. Failed: all coefficient signs match credit logic (7 of 16 do not), and the
challenger does not beat the candidate by 0.01 AUC (it does, by +0.0154). Details are in the README table.

## 8. Findings

| # | Severity | Finding | Recommendation |
|---|---|---|---|
| F1 | Medium | Seven coefficients have the wrong sign because of collinearity among delay variables (VIF up to 18.2). Predictions are fine but the coefficients cannot be explained to a customer or regulator | Adopt the 12-feature specification (no sign problems, no accuracy loss) and revalidate |
| F2 | Medium | The challenger beats the candidate by +0.0154 AUC (interval above zero) | The model owner should document why the logistic model is retained (interpretability, adverse-action reasons) or adopt the challenger with explainability controls |
| F3 | Low to medium | Education coefficients rest on a 345-record reference group, are unstable, add 0.0008 AUC, and may proxy for protected characteristics | Remove education |
| F4 | Low | Weakest ranking where most customers are (not late last month, AUC 0.658). Calibration gaps for three groups at two or more standard errors | Monitor segment AUC and group calibration after deployment |
| F5 | Low | Documentation gaps: undefined repayment codes (17,496 rows), undefined education and marriage codes, 56 duplicates, 590 negative bills | Confirm code meanings with the data owner and document the treatment |
| F6 | Limitation | No time dimension, so drift and out-of-time performance could not be tested | Require an out-of-time test and a PSI monitoring plan before any real use |

## 9. Conclusion and conditions of use

The candidate ranks customers well (AUC 0.775, KS 0.418), is well calibrated overall, is stable under
resampling and input noise, and passes the fairness screen. It is conditionally acceptable for ranking
customers on this data, subject to resolving F1 and F3 and documenting F2. It should not be used
operationally until F6 is addressed, because nothing here shows how it performs on a later period.

## 10. Limitations of this validation

- Single dataset; the model is not a production credit model
- Self-validation (see the independence note)
- Thresholds are rules of thumb, not a policy
- The test set was used for the final comparisons, including the challenger and the simpler specification
- No reject inference, no legal or regulatory compliance assessment
- Group-level results rest on small samples for some groups

## 11. Reproduce

```bash
pip install numpy pandas scikit-learn scipy matplotlib
python validate_credit_model.py
```
