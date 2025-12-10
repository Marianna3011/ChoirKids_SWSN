# Cold-Start Analysis in GNNs: Applying LiGNN to Product Graphs

**Authors:** Jakub Liszyński 156060, Marianna Myszkowska 156041  
**Course:** Special Ways in Software Engineering | December 2025

---

## Abstract

We apply GraphSAGE (inspired by LinkedIn's LiGNN) to analyze cold-start performance on OGB ogbn-products. Cold-start nodes (degree ≤ 2, 9.6% of graph) show 14% accuracy drop compared to well-connected nodes (89% vs 75%), revealing significant challenges when transferring GNN methods from social networks to product graphs.

---

## 1. Introduction

LinkedIn's LiGNN framework addresses cold-start in billion-node social graphs. We test whether these methods (GraphSAGE + multi-hop sampling) generalize to product recommendation graphs using OGB ogbn-products (2.45M nodes, 61M edges). **Contribution:** First reproducible cold-start benchmark quantifying GNN performance on low-degree product nodes.

---

## 2. Related Work

- **LiGNN** [LinkedIn, 2020]: Production GNN with graph densification and multi-hop sampling for cold-start
- **GraphSAGE** [Hamilton+, 2017]: Scalable GNN via neighborhood sampling
- **GAT** [Veličković+, 2018]: Attention mechanisms for weighted aggregation
- **OGB** [Hu+, 2020]: Open benchmark datasets

---

## 3. Dataset & Cold-Start Definition

**OGB ogbn-products**: Amazon co-purchasing network with 2.45M products, 61.9M edges, 100-dim features, 47 categories.

**Cold-Start:** Nodes with degree ≤ 2 (236,337 products = 9.6%). Test set contains 235,990 cold-start nodes (10.7% of test).

---

## 4. Method

**Baseline MLP:** 100 → 256 → 47 (features only, no graph). Val accuracy: 27.29%

**GraphSAGE:** 2-layer mean aggregation, 256 hidden units, 75K params. Multi-hop sampling: 10 neighbors (1-hop) + 5 neighbors per 1-hop (2-hop) = max 61 nodes per target. Training: batch 1024, Adam lr=0.001, 10 epochs.

---

## 5. Results

| Model | Val (All) | Val (Cold) | Test (All) | Test (Cold) | Gap |
|-------|-----------|------------|------------|-------------|-----|
| MLP | 27.29% | — | — | — | — |
| **GraphSAGE** | **89.27%** | 74.64% | **74.92%** | 65.11% | **14%** |

**Findings:**
1. Graph structure critical: GraphSAGE beats MLP by 62 points
2. Cold-start gap: 14-15% accuracy drop for degree ≤ 2 nodes  
3. Multi-hop partial help: 65-75% accuracy shows compensation but not solution
4. 236K test cold-start nodes represent real deployment challenge

---

## 6. Discussion

**Why the gap?** Limited neighbors reduce information flow; sampling becomes ineffective.

**Why it works partially?** 2-hop neighborhoods provide extended reach; features offer baseline signal.

**Domain insights:** Product graphs differ from social networks—cold-start more severe than expected from LiGNN results.

---

## 7. Limitations & Future Work

- No attention weighting (uniform sampling)
- Missing metadata integration
- Static graph only

**Next:** GAT for learned neighbor importance, feature augmentation, graph densification.

---

## 8. Conclusions

Applied LiGNN methods to product graphs: GraphSAGE scales well but cold-start remains challenging (14% gap). First reproducible benchmark on ogbn-products enables future cold-start research.

---

## References

1. LinkedIn Blog. "Graph Neural Networks at LinkedIn" (2020)
2. Hamilton+ "Inductive Representation Learning on Large Graphs." NeurIPS 2017
3. Kipf & Welling. "Semi-Supervised Classification with GCNs." ICLR 2017
4. Veličković+ "Graph Attention Networks." ICLR 2018
5. Hu+ "Open Graph Benchmark." NeurIPS 2020
6. Ying+ "Graph CNNs for Web-Scale Recommender Systems." KDD 2018

**Code:** https://github.com/Marianna3011/ChoirKids_SWSN
