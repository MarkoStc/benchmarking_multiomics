# Benchmarking Multi-Omics Classification

This repository benchmarks multi-omics classification methods across three TCGA datasets:

- **TCGA-BRCA**
- **TCGA-KIPAN**
- **TCGA-LGG**

The benchmark studies not only overall predictive performance, but also how performance changes with:

1. the number of available patients;
2. the number and identity of omics;
3. block-wise omic missingness;
4. the percentage of retained features within each omic.

All reported performance values must include a measure of variability. Unless stated otherwise, results are reported as **mean ± standard deviation (SD)** across matched outer resampling units.

---

## 1. Benchmark objectives

The benchmark must answer the following questions for each of the three datasets.

1. Which method performs best under the standard full-data setting?
2. How much data is required for stable performance?
3. Does adding more omics improve classification?
4. For each number of omics, which exact combination is best?
5. Which methods are most robust when entire patient–omic blocks are missing?
6. How does performance change as the percentage of features retained per omic changes?
7. Which classes remain difficult even when aggregate performance is high?
8. Are predicted probabilities well calibrated?
9. How variable are the conclusions across data splits, random seeds, and missingness masks?

---

## 2. Datasets and tasks

The benchmark is run independently on:

| Dataset | Prediction target | Number of classes | Available omics | Number of eligible patients | Class distribution |
|---|---|---:|---|---:|---|
| TCGA-BRCA | Dataset-specific clinical or molecular class, including PAM50 in the BRCA benchmark | Auto-filled | Auto-filled | Auto-filled | Auto-filled |
| TCGA-KIPAN | Dataset-specific tumour or molecular class | Auto-filled | Auto-filled | Auto-filled | Auto-filled |
| TCGA-LGG | Dataset-specific clinically relevant or molecular class | Auto-filled | Auto-filled | Auto-filled | Auto-filled |

The canonical omic set should be read from the dataset configuration. Where available, the five benchmark views are expected to include:

- mRNA expression;
- DNA methylation;
- microRNA expression;
- copy-number variation;
- RPPA/proteomics.

The exact view set may differ by dataset. The analysis must never silently compare a five-view experiment in one dataset with a four-view experiment in another without displaying the available views.

### Required dataset-description outputs

For every dataset, save:

- total number of eligible patients;
- number of patients per class;
- class proportions;
- number of raw features per omic;
- number of patients observed in each omic;
- number and percentage of complete-case patients;
- patient–omic availability matrix;
- number of patients available for every omic combination.

---

## 3. Evaluation protocol

### 3.1 Patient-level splitting

All splitting must occur at the **patient level**. No samples, aliquots, visits, or repeated measurements from the same patient may appear in both training and test data.

Use stratified outer resampling whenever the class counts allow it. Hyperparameters, feature selection, preprocessing, latent-dimension selection, decision-threshold selection, and omic-combination selection must be performed using training data only.

### 3.2 Matched comparisons

For a given dataset and experiment condition:

- use exactly the same patient subsets for all methods;
- use exactly the same outer train/test splits for all methods;
- use exactly the same missingness masks for all methods;
- use exactly the same feature-percentage condition for all methods;
- store the split ID, repeat ID, fold ID, seed, and mask ID in every result row.

This makes comparisons paired and prevents variation in the sampled patients from being confused with variation between methods.

### 3.3 Nested model selection

Use nested cross-validation or an equivalent nested resampling procedure:

- **outer resampling:** estimates generalization performance;
- **inner cross-validation:** tunes hyperparameters and selects preprocessing settings;
- **outer test set:** used once for final evaluation.

Recommended default when class counts permit:

- 5 outer folds;
- 5 repeats;
- 3 inner folds.

For very small patient subsets, reduce the number of folds only when required to keep every training and test split valid. Record the effective number of folds. Conditions that cannot contain every required class must be marked as unavailable rather than forced.

### 3.4 Aggregation and standard deviations

Do not compute SD across individual patients.

For repeated cross-validation:

1. compute each metric separately in every outer fold;
2. average the outer folds within each repeat;
3. report the mean and SD across repeat-level averages.

Therefore:

\[
\text{reported mean} = \operatorname{mean}_{r}(m_r),
\qquad
\text{reported SD} = \operatorname{SD}_{r}(m_r),
\]

where \(m_r\) is the mean outer-fold score in repeat \(r\).

For experiments with repeated missingness masks or repeated patient subsamples, retain both sources of variation. Report:

- mean performance;
- total SD across independent repeat/mask or repeat/subsample units;
- number of evaluation units;
- optionally, a 95% bootstrap confidence interval.

Every plot must show mean performance and either:

- an SD ribbon;
- SD error bars;
- or individual repeat points plus the mean.

---

## 4. Classification metrics

Because the datasets can be multiclass and imbalanced, **accuracy alone is insufficient**.

### 4.1 Primary metrics

These metrics must be used for the main conclusions.

#### Balanced accuracy

\[
\text{Balanced accuracy}
=
\frac{1}{C}\sum_{c=1}^{C}
\frac{TP_c}{TP_c+FN_c}
\]

Balanced accuracy is the mean recall across classes and gives each class equal importance.

**Use:** primary metric for model ranking, hyperparameter selection, and best-omic-combination selection.

#### Macro F1 score

\[
F1_c =
\frac{2\,\text{Precision}_c\,\text{Recall}_c}
{\text{Precision}_c+\text{Recall}_c},
\qquad
\text{Macro F1} =
\frac{1}{C}\sum_{c=1}^{C}F1_c
\]

Macro F1 gives every class equal weight while requiring both precision and recall to be high.

**Use:** co-primary metric in all main figures and tables.

### 4.2 Required secondary metrics

Calculate these for every outer test prediction.

| Metric | Averaging | Why it is required |
|---|---|---|
| Accuracy | Overall | Familiar global performance measure |
| Macro precision | Macro | Equal-weight precision across classes |
| Macro recall | Macro | Equivalent to balanced accuracy in standard multiclass classification; retained for clarity |
| Weighted F1 | Support-weighted | Describes performance weighted by class prevalence |
| Multiclass MCC | Multiclass | Robust single-number summary using the full confusion matrix |
| Cohen’s kappa | Multiclass | Agreement beyond chance |
| ROC-AUC | Macro one-vs-rest | Ranking quality across classes |
| ROC-AUC | Weighted one-vs-rest | Prevalence-weighted ranking quality |
| PR-AUC / average precision | Macro one-vs-rest | Especially informative for under-represented classes |
| Log loss | Multiclass | Quality and confidence of predicted probabilities |
| Multiclass Brier score | Multiclass | Probability accuracy and calibration |
| Expected calibration error | Multiclass | Difference between confidence and observed correctness |

For methods that do not naturally output probabilities, use calibrated probabilities fitted within the training data only. Never calibrate on the outer test set.

### 4.3 Required per-class metrics

For every class, report:

- number of test observations;
- prevalence;
- precision;
- recall/sensitivity;
- specificity using one-vs-rest coding;
- F1 score;
- one-vs-rest ROC-AUC;
- one-vs-rest PR-AUC;
- predicted-class frequency.

### 4.4 Required confusion matrices

For the best final model in each dataset, save:

1. the raw-count confusion matrix;
2. the confusion matrix normalized by true class;
3. the confusion matrix normalized by predicted class.

The true-class-normalized version is the main report figure because each row shows class-specific recall.

### 4.5 Metric hierarchy for decisions

Use the following fixed hierarchy to avoid choosing different winners with whichever metric is most favourable:

1. highest mean balanced accuracy;
2. highest mean macro F1;
3. highest mean macro PR-AUC;
4. lower log loss;
5. lower model complexity or runtime.

All other metrics are supporting outcomes and must not replace the predefined primary metric after results are observed.

---

## 5. Main full-data benchmark

Run every registered method on every dataset using the complete predefined benchmark setting.

The main benchmark should compare, where implemented:

- single-omic baselines;
- early-integration classifiers;
- late-integration classifiers;
- PCA-based integration;
- MOFA-based integration;
- IntegrAO;
- any additional registered multi-omics method.

All preprocessing and feature selection must occur within the training data.

### Main benchmark figures

#### Figure 1 — Overall performance across datasets

For each dataset:

- x-axis: method;
- y-axis: metric;
- separate panels for balanced accuracy and macro F1;
- bars or points: mean;
- error bars: ±1 SD;
- optionally show individual repeat-level points.

A supplementary version should show accuracy, MCC, macro ROC-AUC, macro PR-AUC, log loss, and Brier score.

#### Figure 2 — Per-class performance

For each dataset and selected top methods:

- heatmap of per-class F1;
- heatmap of per-class recall;
- heatmap of per-class PR-AUC.

#### Figure 3 — Confusion matrices

One true-class-normalized confusion matrix for the selected final model of each dataset.

#### Figure 4 — Calibration

For the selected final model of each dataset:

- multiclass reliability diagram;
- confidence histogram;
- ECE, Brier score, and log loss in the caption or accompanying table.

### Main benchmark tables

#### Table 1 — Dataset characteristics

Columns:

- dataset;
- prediction target;
- class;
- class count;
- class percentage;
- total eligible patients;
- omic name;
- observed patients;
- raw feature count;
- complete-case count.

#### Table 2 — Main model performance

One row per dataset and method.

Required columns:

- dataset;
- method;
- integration type;
- omics used;
- number of omics;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- accuracy, mean ± SD;
- MCC, mean ± SD;
- macro ROC-AUC, mean ± SD;
- macro PR-AUC, mean ± SD;
- log loss, mean ± SD;
- Brier score, mean ± SD;
- ECE, mean ± SD;
- number of outer evaluation units.

#### Table 3 — Per-class results

One row per dataset, method, and class.

Required columns:

- dataset;
- method;
- class;
- support;
- prevalence;
- precision, mean ± SD;
- recall, mean ± SD;
- specificity, mean ± SD;
- F1, mean ± SD;
- ROC-AUC, mean ± SD;
- PR-AUC, mean ± SD.

---

## 6. Experiment A: performance versus number of patients

### 6.1 Patient-count grid

Use the following requested total-patient grid:

\[
N \in \{20, 50, 100, 200, 300, 400, 500\}.
\]

Also include the maximum available cohort size as a final **Full cohort** condition when it differs from 500.

For each dataset and patient count:

1. draw a stratified patient subset;
2. preserve class proportions as closely as possible;
3. use matched subsets across all methods;
4. use nested patient subsets where practical, so the patients at a smaller \(N\) are contained in the larger-\(N\) subset for the same seed;
5. repeat subsampling across multiple seeds;
6. run the same nested evaluation protocol.

If a requested \(N\) exceeds the eligible cohort size, record the condition as unavailable.

### 6.2 Patient-count figures

#### Figure 5 — Primary metrics versus number of patients

For each dataset:

- x-axis: number of patients;
- y-axis: balanced accuracy or macro F1;
- one line per method;
- mean across repeated subsamples/resampling;
- shaded ribbon: ±1 SD;
- include the full-cohort point;
- use the same y-axis range across datasets when this remains readable.

Balanced accuracy and macro F1 should be shown as separate panels or separate figures.

#### Figure 6 — Stability versus number of patients

For each dataset and method:

- x-axis: number of patients;
- y-axis: SD of balanced accuracy across repeats;
- lower values indicate more stable estimates.

#### Figure 7 — Learning-curve efficiency

Optional but recommended:

- x-axis: number of patients;
- y-axis: improvement in balanced accuracy relative to the previous patient-count condition;
- identify where performance begins to plateau.

### 6.3 Patient-count table

#### Table 4 — Patient-count sweep

One row per dataset, method, and patient-count condition.

Required columns:

- dataset;
- method;
- requested number of patients;
- actual number of patients;
- class counts in the sampled cohort;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- accuracy, mean ± SD;
- MCC, mean ± SD;
- macro ROC-AUC, mean ± SD;
- macro PR-AUC, mean ± SD;
- log loss, mean ± SD;
- number of subsampling seeds;
- number of valid outer evaluations.

---

## 7. Experiment B: number and combination of omics

### 7.1 Evaluate every available combination

For \(M\) available omics, evaluate every non-empty subset:

\[
\sum_{k=1}^{M}\binom{M}{k}=2^M-1.
\]

With five omics, this gives 31 combinations.

For every dataset, method, and number of omics \(k\):

- evaluate all \(\binom{M}{k}\) combinations;
- retain the performance of every fixed combination;
- identify the best combination using the predefined metric hierarchy;
- calculate how often each combination wins across repeat-level evaluations.

### 7.2 Avoiding optimistic combination selection

Two outputs are required.

#### Fixed-combination analysis

Every combination is evaluated as a predefined model configuration. The table shows its mean outer-test performance. The combination with the highest mean balanced accuracy is called the **best fixed combination**.

This is useful for interpretation but is a post-hoc comparison across configurations.

#### Nested selected-combination analysis

Treat the omic combination as a hyperparameter:

1. compare combinations using inner-CV balanced accuracy;
2. select one combination independently in each outer training split;
3. evaluate that selected combination on the outer test split;
4. report outer-test mean ± SD;
5. report how frequently every combination was selected.

This is the unbiased estimate of a procedure that selects the best omic combination from training data.

### 7.3 Best-combination rules

For each dataset, method, and \(k\), rank combinations by:

1. mean balanced accuracy;
2. mean macro F1;
3. mean macro PR-AUC;
4. lower log loss;
5. lower number of retained features or lower runtime.

Also report:

- difference from the second-best combination;
- rank SD;
- winner frequency;
- whether the top combinations are practically indistinguishable within one SD.

### 7.4 Omic-number and combination figures

#### Figure 8 — Best performance versus number of omics

For each dataset:

- x-axis: number of omics \(k\);
- y-axis: balanced accuracy or macro F1;
- one line per method;
- at each \(k\), plot the best fixed omic combination;
- error ribbon or error bar: ±1 SD;
- label or annotate the winning omic combination at every \(k\).

Create separate balanced-accuracy and macro-F1 versions.

#### Figure 9 — Best omic combination at each \(k\)

For each dataset and method:

- x-axis: \(k\);
- y-axis: best combination;
- cell text: mean balanced accuracy ± SD;
- cell annotation: winner frequency.

A compact alternative is a labelled dot plot.

#### Figure 10 — All omic combinations

For each dataset:

- x-axis: exact omic combination;
- y-axis: balanced accuracy;
- points or bars: mean;
- error bars: ±1 SD;
- group or facet by number of omics;
- order combinations by mean balanced accuracy within each \(k\).

#### Figure 11 — Omic contribution summary

Recommended:

- heatmap of how often each omic appears in the best-performing combination;
- calculate separately for each dataset, method, and \(k\).

This is descriptive and does not replace the all-combination analysis.

### 7.5 Omic-combination tables

#### Table 5 — Best combination for each number of omics

One row per dataset, method, and \(k\).

Required columns:

- dataset;
- method;
- number of available omics;
- number of omics used;
- best fixed omic combination;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- macro PR-AUC, mean ± SD;
- second-best combination;
- balanced-accuracy difference from second best;
- winner frequency;
- mean rank;
- rank SD;
- number of evaluated combinations.

#### Table 6 — Complete omic-combination leaderboard

One row per dataset, method, and exact combination.

Required columns:

- dataset;
- method;
- omic combination;
- number of omics;
- patient count available for this combination;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- accuracy, mean ± SD;
- MCC, mean ± SD;
- macro ROC-AUC, mean ± SD;
- macro PR-AUC, mean ± SD;
- log loss, mean ± SD;
- mean rank;
- rank SD;
- winner frequency.

#### Table 7 — Nested combination selection

One row per dataset, method, and \(k\).

Required columns:

- dataset;
- method;
- number of omics;
- outer-test balanced accuracy, mean ± SD;
- outer-test macro F1, mean ± SD;
- most frequently selected combination;
- selection frequency;
- entropy or diversity of selected combinations;
- number of outer evaluations.

---

## 8. Experiment C: omic missingness

### 8.1 Missingness grid

Use the requested grid:

\[
p_{\text{missing}} \in \{0, 25, 50, 75, 100\}\%.
\]

Missingness must be applied at the **patient–omic block level**, not by randomly deleting individual feature values.

### 8.2 Two required missingness settings

#### Random block missingness

Randomly mask the requested percentage of otherwise observed patient–omic blocks.

Constraints:

- use the same masks for every method;
- apply masks independently within the training and test partitions according to the predefined experiment;
- guarantee that each evaluated patient retains at least one observed omic;
- repeat every missingness level with multiple mask seeds;
- preserve labels and patient splits.

Because all omics cannot be removed from a patient, “100% random global missingness” is not mathematically valid. Therefore, the 100% condition in this setting means the maximum feasible block missingness while retaining at least one observed omic per patient, and the achieved missingness percentage must be reported.

#### Complete absence of a designated omic

For each omic separately, remove that omic from every patient. This is the interpretable **100% missingness for one omic** condition.

Also evaluate leave-two-omics-out conditions where computationally feasible.

### 8.3 Missingness figures

#### Figure 12 — Performance versus random omic missingness

For each dataset:

- x-axis: achieved percentage of missing patient–omic blocks;
- y-axis: balanced accuracy or macro F1;
- one line per method;
- mean across split and mask repetitions;
- ribbon: ±1 SD.

#### Figure 13 — Relative performance degradation

For each dataset and method:

\[
\Delta m(p)=m(p)-m(0).
\]

Plot:

- x-axis: missingness percentage;
- y-axis: change from the no-added-missingness baseline;
- show ±1 SD.

This separates absolute model quality from robustness.

#### Figure 14 — Completely missing individual omics

For each dataset:

- x-axis: omitted omic;
- y-axis: balanced accuracy;
- one point or bar per method;
- mean ± SD;
- include the no-omission baseline.

#### Figure 15 — Missingness robustness summary

For each method and dataset, calculate the normalized area under the performance-versus-missingness curve. Higher values indicate greater robustness.

### 8.4 Missingness tables

#### Table 8 — Random block-missingness results

One row per dataset, method, and missingness level.

Required columns:

- dataset;
- method;
- requested missingness percentage;
- achieved missingness percentage;
- mean observed omics per patient;
- percentage of patients with exactly 1, 2, ..., \(M\) observed omics;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- accuracy, mean ± SD;
- MCC, mean ± SD;
- macro PR-AUC, mean ± SD;
- log loss, mean ± SD;
- performance change from 0% missingness, mean ± SD;
- number of mask seeds;
- number of valid outer evaluations.

#### Table 9 — Complete omic-absence results

One row per dataset, method, and omitted omic.

Required columns:

- dataset;
- method;
- omitted omic;
- remaining omics;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- change in balanced accuracy from baseline, mean ± SD;
- change in macro F1 from baseline, mean ± SD;
- macro PR-AUC, mean ± SD;
- log loss, mean ± SD.

#### Table 10 — Missingness robustness ranking

One row per dataset and method.

Required columns:

- dataset;
- method;
- baseline balanced accuracy;
- balanced accuracy at 25%, 50%, and 75% missingness;
- drop at 75%;
- normalized area under the missingness curve;
- robustness rank;
- rank SD.

---

## 9. Experiment D: percentage of features per omic

### 9.1 Feature-percentage grid

Use the requested percentages:

\[
\{0.001,\ 0.01,\ 0.1,\ 1,\ 2,\ 5,\ 10,\ 20,\ 25,\ 50,\ 75,\ 100\}\%.
\]

For each omic, convert the percentage to an integer feature count using:

\[
n_{\text{features}} =
\max\left(1,\left\lceil
p\,D/100
\right\rceil\right),
\]

where \(D\) is the number of available features in that omic after basic quality control.

Always report both the requested percentage and the resulting number of retained features for every omic.

### 9.2 Feature-selection rules

Feature ranking and selection must be fitted using the outer-training data only.

Permitted rankings should be predefined, for example:

- variance-based ranking;
- univariate class association such as ANOVA effect size;
- model-specific selection fitted within inner CV.

Do not rank features once using the full dataset.

### 9.3 Two feature-percentage analyses

#### Joint percentage sweep

Apply the same percentage to every included omic. This is the main feature-scaling experiment and keeps the x-axis interpretable.

#### One-omic-at-a-time sweep

For each omic:

- vary the percentage retained in that omic;
- keep all other omics at 100%;
- quantify which omics are sensitive to aggressive feature reduction.

A full Cartesian product of percentages across five omics is not required because it is prohibitively large.

### 9.4 PCA and MOFA settings

For PCA- and MOFA-based methods:

1. apply the same training-only feature prefilter used in the feature-percentage condition;
2. tune the latent representation within the inner CV;
3. record the number of components/factors;
4. where applicable, evaluate predefined explained-variance targets.

Store:

- requested feature percentage;
- feature count per omic;
- selected PCA variance target;
- selected number of PCA components;
- selected MOFA factor count or model-selection setting.

### 9.5 Feature-percentage figures

#### Figure 16 — Performance versus percentage of features

For each dataset:

- x-axis: percentage of features retained per omic;
- use a logarithmic x-axis;
- y-axis: balanced accuracy or macro F1;
- one line per method;
- mean ± SD ribbon;
- annotate the resulting feature counts in a supplementary version.

#### Figure 17 — Performance versus total retained feature count

For each dataset:

- x-axis: total number of retained features across omics;
- logarithmic x-axis;
- y-axis: balanced accuracy;
- one line per method;
- mean ± SD.

This complements the percentage plot because the omics have different raw dimensionalities.

#### Figure 18 — One-omic-at-a-time feature sensitivity

For each dataset:

- one panel per omic;
- x-axis: percentage retained in that omic;
- y-axis: change in balanced accuracy from the 100% condition;
- other omics fixed at 100%;
- one line per method;
- mean ± SD.

#### Figure 19 — Best feature-efficiency point

Plot balanced accuracy against total retained features. Mark Pareto-efficient configurations that cannot improve performance without using more features.

### 9.6 Feature-percentage tables

#### Table 11 — Joint feature-percentage sweep

One row per dataset, method, and feature-percentage condition.

Required columns:

- dataset;
- method;
- feature percentage;
- retained features in each omic;
- total retained features;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- accuracy, mean ± SD;
- MCC, mean ± SD;
- macro PR-AUC, mean ± SD;
- log loss, mean ± SD;
- selected latent dimension, mean ± SD where applicable;
- number of valid outer evaluations.

#### Table 12 — One-omic-at-a-time feature sensitivity

One row per dataset, method, varied omic, and feature percentage.

Required columns:

- dataset;
- method;
- varied omic;
- feature percentage in varied omic;
- retained feature count in varied omic;
- percentages in other omics;
- balanced accuracy, mean ± SD;
- macro F1, mean ± SD;
- change from the 100% baseline, mean ± SD.

#### Table 13 — Feature-efficiency summary

One row per dataset and method.

Required columns:

- dataset;
- method;
- best balanced accuracy;
- percentage at best balanced accuracy;
- smallest percentage within one SD of the best;
- total features at that point;
- macro F1 at that point;
- feature-efficiency rank.

The “smallest percentage within one SD of the best” is the recommended compact configuration.

---

## 10. Cross-dataset summary

The three datasets must first be analysed separately. Do not pool patient-level predictions across datasets.

### Cross-dataset figures

#### Figure 20 — Method rank across datasets

For every dataset:

- rank methods by balanced accuracy;
- show the mean rank across TCGA-BRCA, TCGA-KIPAN, and TCGA-LGG;
- also display each dataset-specific rank.

Because there are only three datasets, treat this as a descriptive summary rather than strong inferential evidence.

#### Figure 21 — Robustness profile

For each method, summarize standardized or rank-based performance on:

- full-data balanced accuracy;
- patient efficiency;
- omic-number performance;
- missingness robustness;
- feature efficiency;
- calibration.

Use a heatmap rather than a radar chart where possible.

### Cross-dataset tables

#### Table 14 — Cross-dataset method summary

One row per method.

Required columns:

- method;
- mean balanced accuracy rank;
- mean macro F1 rank;
- mean missingness-robustness rank;
- mean feature-efficiency rank;
- number of dataset wins;
- average runtime if measured;
- datasets successfully completed.

#### Table 15 — Dataset-specific winners

One row per dataset and benchmark question.

Required columns:

- dataset;
- benchmark question;
- winning method;
- winning omic combination;
- number of omics;
- condition;
- primary metric, mean ± SD;
- runner-up;
- difference from runner-up.

Benchmark questions include:

- best full-data performance;
- best at smallest patient count;
- best at 50% missingness;
- most robust across missingness;
- best compact feature configuration;
- best single omic;
- best two-omic combination;
- best three-omic combination;
- best four-omic combination;
- best five-omic combination.

---

## 11. Statistical comparisons

Performance comparisons should be paired because methods use matched patient splits and masks.

For every dataset and primary comparison:

1. calculate the repeat-level difference in balanced accuracy and macro F1;
2. report the mean paired difference;
3. report its SD;
4. report a 95% bootstrap confidence interval over repeat-level units;
5. optionally use a paired permutation test or Wilcoxon signed-rank test;
6. correct families of comparisons using Benjamini–Hochberg FDR.

Do not base the main conclusion only on a p-value. Report effect size and uncertainty.

### Required comparison table

#### Table 16 — Pairwise method comparisons

Columns:

- dataset;
- experiment;
- condition;
- method A;
- method B;
- metric;
- mean paired difference;
- SD of paired difference;
- 95% confidence interval;
- raw p-value;
- FDR-adjusted p-value;
- number of paired units.

---

## 12. Computational and reproducibility outputs

Recommended for the supplementary report:

### Figure 22 — Performance versus computational cost

- x-axis: median training time or total benchmark time;
- y-axis: balanced accuracy;
- point size may represent peak memory if a static plotting implementation supports it;
- one point per method and dataset.

### Table 17 — Computational cost

Columns:

- dataset;
- method;
- experiment;
- condition;
- training time, mean ± SD;
- inference time, mean ± SD;
- peak CPU memory;
- peak GPU memory;
- number of trainable parameters;
- hardware identifier.

### Reproducibility metadata

Every run must store:

- dataset version;
- task name;
- omics;
- patient IDs or split-manifest hash;
- outer repeat and fold;
- inner-CV configuration;
- random seed;
- missingness-mask seed;
- feature-ranking method;
- selected features or feature-list hash;
- preprocessing configuration;
- model hyperparameters;
- software environment;
- git commit;
- run status and error message.

---

## 13. Required result-file schema

### Prediction-level file

Save one row per outer-test patient and run:

```text
dataset
task
method
integration_type
omic_combination
n_omics
experiment
condition
patient_id
true_label
predicted_label
probability_<class_1>
...
probability_<class_C>
repeat_id
outer_fold
seed
subsample_id
missingness_mask_id
feature_percentage
split_manifest_hash
```

### Fold-level metric file

Save one row per outer fold:

```text
dataset
task
method
omic_combination
n_omics
experiment
condition
repeat_id
outer_fold
seed
n_train
n_test
balanced_accuracy
macro_f1
accuracy
macro_precision
macro_recall
weighted_f1
mcc
cohen_kappa
macro_roc_auc_ovr
weighted_roc_auc_ovr
macro_pr_auc_ovr
log_loss
brier_score
ece
runtime_seconds
status
```

### Aggregated result file

Save one row per plotted point or report-table row:

```text
dataset
method
experiment
condition
omic_combination
n_omics
metric
mean
std
n_units
ci95_low
ci95_high
```

Per-class results should use a separate long-format file containing `class_name`, `metric`, `mean`, and `std`.

---

## 14. Required output structure

```text
results/
├── dataset_characteristics/
│   ├── TCGA_BRCA_characteristics.csv
│   ├── TCGA_KIPAN_characteristics.csv
│   └── TCGA_LGG_characteristics.csv
├── predictions/
├── fold_metrics/
├── aggregated_metrics/
├── omic_combinations/
├── patient_sweep/
├── missingness/
├── feature_sweep/
├── statistical_tests/
└── computational_cost/

figures/
├── main/
│   ├── fig01_overall_performance.*
│   ├── fig02_per_class_performance.*
│   ├── fig03_confusion_matrices.*
│   ├── fig05_metrics_vs_patients.*
│   ├── fig08_best_performance_vs_n_omics.*
│   ├── fig09_best_combination_by_n_omics.*
│   ├── fig12_metrics_vs_missingness.*
│   └── fig16_metrics_vs_feature_percentage.*
└── supplementary/
    ├── all_metrics/
    ├── all_omic_combinations/
    ├── calibration/
    ├── learning_curve_stability/
    ├── missing_omic_identity/
    ├── feature_sensitivity/
    └── computational_cost/

tables/
├── table01_dataset_characteristics.csv
├── table02_main_model_performance.csv
├── table03_per_class_results.csv
├── table04_patient_count_sweep.csv
├── table05_best_combination_by_n_omics.csv
├── table06_all_omic_combinations.csv
├── table07_nested_combination_selection.csv
├── table08_random_missingness.csv
├── table09_complete_omic_absence.csv
├── table10_missingness_robustness.csv
├── table11_feature_percentage_sweep.csv
├── table12_per_omic_feature_sensitivity.csv
├── table13_feature_efficiency.csv
├── table14_cross_dataset_summary.csv
├── table15_dataset_winners.csv
├── table16_pairwise_comparisons.csv
└── table17_computational_cost.csv
```

Save report figures in both a vector format (`.pdf` or `.svg`) and a high-resolution raster format (`.png`).

---

## 15. Minimum main-report set

The main paper or report should include at least:

### Main figures

1. Overall balanced accuracy and macro F1 across methods and datasets.
2. Balanced accuracy and macro F1 versus number of patients.
3. Best performance versus number of omics, labelled with the best omic combination at each \(k\).
4. Performance versus omic missingness.
5. Performance versus percentage of retained features.
6. Confusion matrix for the selected final model in each dataset.

### Main tables

1. Dataset characteristics.
2. Overall model performance with mean ± SD.
3. Best omic combination for every number of omics.
4. Missingness robustness summary.
5. Feature-efficiency summary.
6. Per-class results for the selected final models.

Everything else should be retained in supplementary material rather than discarded.

---

## 16. Primary conclusions to extract

For every dataset, the final report must explicitly state:

- the best method by balanced accuracy;
- its macro F1 and uncertainty;
- the best single omic;
- the best two-omic combination;
- the best three-omic combination;
- the best four-omic combination where available;
- the best five-omic combination where available;
- whether adding an omic improves performance beyond variability;
- the minimum patient count at which performance is within one SD of full-cohort performance;
- the performance loss at 25%, 50%, and 75% missingness;
- the omic whose complete absence causes the largest performance drop;
- the smallest feature percentage within one SD of the best performance;
- the classes with the lowest recall and lowest PR-AUC;
- whether the selected model is adequately calibrated.

---

## 17. Non-negotiable reporting rules

- Always report **mean ± SD** and the number of evaluation units.
- Use **balanced accuracy** as the primary metric.
- Report **macro F1** beside balanced accuracy in every main performance result.
- Never use accuracy alone to choose a model.
- Never perform feature selection before splitting the data.
- Never tune hyperparameters, thresholds, latent dimensions, or omic combinations on the outer test set.
- Never compare methods using different patient splits or missingness masks.
- Never report only the best omic combination; retain the complete combination leaderboard.
- For every number of omics, explicitly name the best combination.
- Distinguish post-hoc best fixed combinations from nested, training-selected combinations.
- Do not interpret 100% global block missingness as removal of every omic from every patient.
- Keep results for TCGA-BRCA, TCGA-KIPAN, and TCGA-LGG separate before producing cross-dataset summaries.
