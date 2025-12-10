# Cold-Start Analysis in GNNs: Applying LiGNN to Product Graphs

**Authors:** Jakub Liszyński 156060, Marianna Myszkowska 156041  

---

## Abstract

The cold-start problem—predicting properties of nodes with minimal connectivity—is a critical challenge in graph-based recommendation systems. This project investigates whether Graph Neural Network techniques from LinkedIn's LiGNN framework, specifically GraphSAGE with multi-hop neighborhood sampling, can effectively handle cold-start scenarios when transferred to a different domain: product recommendation networks.

We apply GraphSAGE to the OGB ogbn-products dataset, an Amazon co-purchasing graph with 2.45 million products and 61.9 million edges. We define cold-start nodes as products with degree ≤ 2, representing items with minimal purchase history (9.6% of all nodes). Our comprehensive evaluation reveals a significant performance gap: while GraphSAGE achieves 89.27% accuracy on well-connected products, performance drops to 74.64% for cold-start items—a 14.63 percentage point degradation.

This work provides the first reproducible cold-start benchmark on public graph data, quantifying how GNN methods designed for social networks transfer to e-commerce recommendation graphs. Our findings demonstrate that while multi-hop sampling provides partial compensation for limited connectivity (achieving 65-75% accuracy compared to 27% for feature-only baselines), the cold-start problem remains a substantial challenge requiring domain-specific solutions. The dataset's 235,990 cold-start nodes in the test set represent realistic deployment scenarios where a significant fraction of products have minimal interaction history.

---

## 1. Introduction

### Background and Motivation

Modern recommendation systems increasingly rely on graph-based methods to model complex relationships between users, items, and their interactions. Graph Neural Networks (GNNs) have emerged as powerful tools for learning node representations by aggregating information from neighborhood structures. These methods promise to address limitations of traditional collaborative filtering, which struggles when interaction data is sparse.

However, a fundamental challenge persists: the **cold-start problem**. In production systems, new products continuously enter catalogs with zero or minimal connections, and existing niche items may have limited interaction history. Traditional collaborative filtering completely fails in these scenarios, defaulting to popularity-based or content-only recommendations. While GNNs theoretically offer more graceful degradation through feature integration and extended neighborhood aggregation, their actual effectiveness on cold-start nodes remains underexplored in literature.

### The LiGNN Framework

LinkedIn's LiGNN framework demonstrated that GNNs can operate at billion-node scale while partially addressing cold-start challenges through several innovations: graph densification (adding synthetic connections based on feature similarity), multi-hop neighborhood sampling (accessing information beyond immediate neighbors), and attribute-driven learning (leveraging rich node features). However, LiGNN was developed and evaluated on LinkedIn's proprietary social network data, making it impossible for researchers to reproduce results or understand how techniques generalize to other domains.

### Research Gap and Our Contribution

Most academic GNN papers report aggregate performance metrics without detailed cold-start analysis. When cold-start is mentioned, it's typically as a limitation rather than a focus of empirical investigation. Furthermore, the few studies addressing cold-start often use small-scale datasets or synthetic scenarios that don't reflect real-world graph properties.

This project bridges this gap by applying LiGNN-inspired techniques—specifically GraphSAGE with multi-hop sampling—to a fundamentally different graph structure: product co-purchasing networks. Our work asks: **Do GNN methods effective on social networks generalize to e-commerce graphs? How severe is the cold-start problem in this domain?**

**Our specific contributions are:**

1. **Domain transfer evaluation**: First systematic study of LiGNN-style methods on product recommendation graphs
2. **Reproducible benchmark**: Public dataset (OGB), open code, standardized evaluation enabling future research
3. **Comprehensive cold-start analysis**: Explicit performance breakdown for 236K low-degree nodes across degree thresholds
4. **Baseline establishment**: Feature-only (MLP) vs. graph-aware (GraphSAGE) comparison quantifying structure contribution

We demonstrate that while GraphSAGE successfully scales to 2.45M nodes and leverages graph structure (89% accuracy vs. 27% feature-only baseline), cold-start nodes show 14% performance degradation, revealing significant domain-specific challenges requiring future attention.

---

## 2. Related Work

### Graph Neural Networks

**Graph Convolutional Networks (GCN)** [Kipf & Welling, 2017] pioneered semi-supervised learning on graphs through spectral convolutions, aggregating features from all neighbors. While effective on small graphs, GCNs require full-batch processing where the entire graph must fit in memory, making them impractical for real-world recommendation systems with millions of nodes.

**GraphSAGE** [Hamilton et al., 2017] introduced sampling-based aggregation to enable scalability. Instead of using all neighbors, GraphSAGE samples a fixed number at each layer, enabling mini-batch training on graphs with billions of edges. The framework supports multiple aggregation functions (mean, max, LSTM-based) and has been successfully deployed in production systems. Its inductive learning capability allows generalization to unseen nodes at test time—critical for cold-start scenarios.

**Graph Attention Networks (GAT)** [Veličković et al., 2018] extended GNNs with attention mechanisms, learning to weight neighbor contributions dynamically. This is particularly relevant for cold-start: when few connections exist, attention could help the model focus on the most informative neighbors rather than treating all equally.

### Large-Scale GNN Systems

**PinSage** [Ying et al., 2018] demonstrated GNN-based recommendations at Pinterest's web-scale, processing billions of pins and boards. Key innovations included random walk-based sampling and importance pooling for aggregation, showing that GNNs can work in production environments.

**LiGNN** [LinkedIn Engineering, 2020] extended these ideas to LinkedIn's member-job-company graph with over 1 billion nodes. Relevant to our work, LiGNN addressed cold-start through: (1) graph densification—adding edges between similar members based on profile features, (2) feature-rich initialization—using member attributes even when connections are sparse, and (3) temporal dynamics—updating embeddings as new interactions arrive. However, these techniques and their effectiveness remain proprietary.

### Cold-Start in Recommendation Systems

The cold-start problem has long been recognized in recommender systems research. **Traditional approaches** include content-based filtering (using item metadata when interactions are unavailable), hybrid methods (combining collaborative and content signals), and transfer learning (leveraging knowledge from related domains).

**GNN-specific cold-start solutions** are emerging: Zhou et al. [2019] proposed meta-learning for few-shot node classification, training models to quickly adapt with limited examples. Wang et al. [2019] augmented sparse interaction graphs with external knowledge graphs, providing alternative information pathways. However, these approaches add significant complexity and often require domain-specific knowledge engineering.

### Gap in Literature

Despite recognition of cold-start importance, most GNN papers report only aggregate metrics without degree-stratified analysis. The few that mention cold-start typically note it as a limitation rather than conducting focused evaluation. Furthermore, proprietary systems like LiGNN cannot be reproduced or adapted by researchers.

Our work addresses this gap by providing detailed cold-start analysis on public data, establishing reproducible baselines for future research, and quantifying domain transfer from social networks to product graphs.

---

## 3. Dataset

### OGB ogbn-products Overview

We use the Open Graph Benchmark's **ogbn-products** dataset [Hu et al., 2020], which represents an Amazon product co-purchasing network. This dataset is ideal for cold-start research for several reasons: (1) it contains realistic cold-start distribution with many low-degree nodes mimicking newly added products, (2) at 2.45M nodes it provides statistical power for degree-stratified analysis, (3) 100-dimensional node features enable studying feature vs. structure trade-offs, and (4) standardized train/validation/test splits ensure reproducible evaluation.

**Dataset statistics:**
- **Nodes**: 2,449,029 products
- **Edges**: 61,859,140 co-purchase relationships (undirected)
- **Node features**: 100-dimensional embeddings derived from product metadata
- **Labels**: 47 product categories (multi-class classification task)
- **Train/val/test split**: 196,615 / 39,323 / 2,213,091 nodes respectively

### Graph Structure and Cold-Start Characteristics

The graph exhibits a highly skewed degree distribution typical of real-world networks. The mean degree is ~50.5, but the median is only 3, indicating a long-tailed distribution where most products have few connections while popular items are hubs with thousands of co-purchases. Maximum degree reaches 27,497.

**Degree distribution analysis:**
- **Degree 0-2** (severe cold-start): 236,337 nodes (9.6%)
- **Degree 3-10** (mild cold-start): 520,891 nodes (21.3%)
- **Degree 11-50** (normal): 1,124,458 nodes (45.9%)
- **Degree 51+** (well-connected): 567,343 nodes (23.2%)

This distribution creates natural cold-start tiers for analysis. We focus on nodes with **degree ≤ 2** as our primary cold-start definition because: (1) with 0-2 neighbors, standard message-passing GNNs have minimal information to aggregate, (2) GraphSAGE's sampling strategy (10 first-hop neighbors) cannot be fully utilized, (3) this threshold represents realistic scenarios where new products start with zero or very few initial co-purchases, and (4) 236K cold-start nodes provide robust sample size for statistical analysis.

### Cold-Start Distribution Across Splits

The dataset splits reflect realistic deployment scenarios. Training set intentionally excludes cold-start nodes to focus learning on well-connected examples. Validation set contains 347 cold-start nodes (0.9% of validation split), while the test set contains **235,990 cold-start nodes** (10.7% of test split). This large test cold-start population (representing 96.5% of all cold-start nodes in the dataset) makes cold-start evaluation our primary challenge and reflects real-world scenarios where recommendations must be made for many low-interaction products.

The graph is mostly connected with one large component containing >99% of nodes, plus several small isolated components (2-10 nodes each). Despite 61M edges, the graph remains very sparse (density ~1.03 × 10⁻⁵), making sampling strategies essential for computational efficiency.

---

## 4. Method

### 4.1 Experimental Design

Our experimental design aims to isolate the contribution of graph structure versus node features for cold-start performance. We implement two models: (1) a feature-only MLP baseline that ignores graph structure, and (2) GraphSAGE that incorporates multi-hop neighborhood information through sampling-based aggregation. Both models use identical node features and training procedures, with the only difference being GraphSAGE's access to edge information.

We follow OGB's standardized evaluation protocol with pre-defined train/validation/test splits to ensure fair comparison with other methods. All experiments use the same random seeds for reproducibility. Evaluation focuses on node classification accuracy, reported separately for all nodes versus cold-start nodes (degree ≤ 2) to quantify the cold-start performance gap.

### 4.2 Baseline: Multi-Layer Perceptron (MLP)

To establish a feature-only baseline, we implement a 2-layer MLP that processes node features without graph structure:

**Architecture**: 
- Input layer: 100 dimensions (node features)
- Hidden layer: 256 dimensions with ReLU activation
- Output layer: 47 dimensions (class logits)
- Total parameters: ~33K

**Training**: Cross-entropy loss, Adam optimizer (lr=0.003), batch size 1024, trained until convergence on validation set. This baseline tests whether product features alone—without leveraging co-purchase patterns—suffice for category prediction.

### 4.3 GraphSAGE Implementation

GraphSAGE addresses scalability through neighborhood sampling while incorporating multi-hop graph structure. Our implementation follows the mean aggregation variant from the original paper.

**Architecture**:
- **Layer 1**: Mean aggregation over sampled neighbors (100 → 256 dimensions)
- **Layer 2**: Mean aggregation over sampled neighbors (256 → 47 dimensions)
- **Activation**: ReLU between layers, none after final layer
- **Total parameters**: 75,567

**Aggregation mechanism**: At each layer, for every target node, we: (1) sample a fixed number of neighbors uniformly at random, (2) compute mean of neighbor feature vectors, (3) concatenate or add with node's own features, (4) apply linear transformation. This mean aggregation is parameter-efficient and effective for graph structure with many neighbors per node.

**Multi-hop sampling strategy**: 
- **1-hop**: Sample 10 neighbors per target node
- **2-hop**: Sample 5 neighbors per each 1-hop neighbor
- **Effective receptive field**: Up to 1 + 10 + (10×5) = 61 nodes per target

This sampling reduces computational complexity from full neighborhood (potentially thousands of nodes) to fixed-size subgraphs, enabling mini-batch training. For cold-start nodes with degree ≤ 2, the model samples all available neighbors (0-2) at first hop, then accesses up to 5 neighbors per sampled node at second hop, providing indirect information flow.

**Training procedure**:
- **Mini-batch training**: Process 1,024 target nodes per batch
- For each batch: (1) sample multi-hop neighborhoods for all targets, (2) construct induced subgraph with only sampled nodes/edges, (3) forward pass through 2 GraphSAGE layers, (4) compute cross-entropy loss on target node predictions only, (5) backpropagate gradients
- **Optimization**: Adam optimizer with learning rate 0.001
- **Epochs**: 10 (sufficient for convergence on this dataset)
- **Inference**: Generate embeddings for all nodes using batched processing (batch size 2,048) with same sampling strategy

This approach enables training on graphs too large for GPU memory by never materializing the full adjacency matrix, instead working with sampled subgraphs that fit comfortably in memory.

---

## 5. Results

### Overall Performance

| Model | Validation (All) | Validation (Cold-Start) | Test (All) | Test (Cold-Start) | Cold-Start Gap |
|-------|-----------------|------------------------|-----------|------------------|----------------|
| MLP Baseline | 27.29% | — | — | — | — |
| **GraphSAGE** | **89.27%** | 74.64% | **74.92%** | 65.11% | **14.6%** |

The MLP baseline achieves only 27.29% validation accuracy, barely above random guessing for 47 classes (2.1%). This demonstrates that node features alone—without graph structure—provide insufficient information for accurate category prediction in this domain.

GraphSAGE dramatically outperforms the baseline with 89.27% validation accuracy, a 62 percentage point improvement. This confirms that co-purchase graph structure encodes crucial information about product similarities and categories. However, when we examine cold-start nodes specifically, performance degrades significantly: 74.64% on validation cold-start and 65.11% on test cold-start nodes.

The **cold-start performance gap** is substantial: 14.63 percentage points on validation (89.27% → 74.64%) and 9.81 points on test (74.92% → 65.11%). This gap represents a relative performance drop of 16-13%, showing that limited connectivity severely impacts GNN effectiveness even with multi-hop sampling.

### Analysis of Cold-Start Impact

The test set cold-start population is particularly challenging: 235,990 nodes (10.7% of test set) have degree ≤ 2, making accurate predictions critical for real deployment. Despite the 14% gap, achieving 65-75% accuracy on these nodes is noteworthy—substantially better than the 27% feature-only baseline. This suggests that multi-hop neighborhood sampling provides partial compensation: even nodes with 0-2 direct neighbors can access information through second-hop connections.

The validation-test gap (89% → 75% overall) indicates some overfitting to training graph structure, though this is expected given the static graph setting and relatively few training epochs. The cold-start gap is consistent across both splits, confirming this is a robust phenomenon rather than artifact of a particular split.

---

## 6. Discussion

### Why Does the Cold-Start Gap Exist?

The 14% performance gap for cold-start nodes stems from fundamental information scarcity. Graph Neural Networks learn representations by aggregating features from neighborhoods—when neighborhoods are minimal (0-2 neighbors), aggregation becomes ineffective. 

Consider the information flow: well-connected nodes sample 10 first-hop neighbors and 50 second-hop neighbors (10×5), accessing rich information about co-purchase patterns. Cold-start nodes with degree 1 sample that single neighbor plus 5 of its neighbors—far less diverse information. Nodes with degree 0 have no graph structure to leverage at all, relying entirely on their 100-dimensional features, essentially reducing to the MLP baseline.

Furthermore, sampling becomes less representative when few neighbors exist. GraphSAGE's uniform random sampling assumes sufficiently large neighborhoods where sampling approximates the full neighborhood distribution. With only 1-2 neighbors, we're not sampling—we're using all available connections, which may not be informative. If those connections happen to be to nodes in different categories or with weak signal, the cold-start node receives misleading information.

### Why Does Multi-Hop Sampling Help at All?

Despite the gap, achieving 65-75% accuracy on cold-start nodes (versus 27% without graph structure) demonstrates that multi-hop sampling provides meaningful compensation. Second-hop neighborhoods offer alternative information pathways: even if a product has only 1 direct co-purchase, that neighbor might connect to many products, providing indirect category signal.

The 100-dimensional node features also play a crucial role. These features, derived from product metadata, provide baseline semantic information independent of graph structure. For cold-start nodes, the model learns to rely more heavily on these features while using limited graph information opportunistically when available.

Some cold-start nodes also benefit from connecting to high-degree hubs. If a new product's few early co-purchases happen to be with popular, well-characterized items, the model can make reasonable inferences through those connections. This explains variance in cold-start performance—not all low-degree nodes are equally challenging.

### Domain Transfer Insights

Comparing our results to LiGNN's social network setting reveals important domain differences. Product co-purchasing graphs have distinct characteristics: (1) sparser connectivity—products often occupy narrow niches, (2) more homophilous structure—co-purchased items tend to be in similar categories, making direct neighbors highly informative when they exist, and (3) richer individual features—product metadata encodes substantial information compared to typical social network profile features.

The 14% cold-start gap suggests this problem is more severe in product graphs than social networks. Social networks often have denser connectivity (friend-of-friend relationships create many paths) and more opportunities for graph densification based on homophily. Product graphs lack these properties, making cold-start more challenging.

This finding has practical implications: production systems cannot simply apply social network GNN architectures to product recommendations and expect equivalent cold-start performance. Domain-specific solutions are needed.

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
