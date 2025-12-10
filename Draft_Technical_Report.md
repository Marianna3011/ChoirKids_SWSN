# Scalable Graph Neural Networks for Large-Scale Recommendation Tasks: A LiGNN-Inspired Approach

**Authors:** Jakub Liszyński, Marianna Myszkowska  
**Date:** December 2025  
**Course:** Special Ways in Software Engineering

---

## Abstract

This project explores scalable Graph Neural Network (GNN) techniques inspired by LinkedIn's LiGNN framework, applied to large-scale product recommendation tasks. Using the OGB ogbn-products dataset (2.45M nodes, 61.9M edges), we implement and evaluate GraphSAGE with multi-hop sampling strategies. Our primary focus is analyzing the cold-start problem—nodes with limited connectivity—which is critical for real-world recommendation systems. Results show that GraphSAGE achieves 89.27% validation accuracy on well-connected nodes but experiences a 14.63 percentage point drop on cold-start nodes, highlighting the challenge and motivating future improvements through attention mechanisms and feature augmentation.

---

## 1. Introduction

### 1.1 Motivation

Modern recommendation systems operate on massive graphs where users, items, and their interactions form complex networks with billions of edges. Traditional collaborative filtering and content-based methods struggle with:

- **Scalability**: Full-batch graph operations become computationally infeasible on large graphs
- **Cold-start problem**: New items or users with few connections lack sufficient signal for accurate predictions
- **Graph structure exploitation**: Many methods ignore the rich relational information encoded in graph topology

Graph Neural Networks (GNNs) offer a promising solution by learning node representations through neighborhood aggregation, naturally incorporating both node features and graph structure. However, applying GNNs to production-scale graphs presents significant engineering and algorithmic challenges.

### 1.2 Problem Statement

This project addresses the following research questions:

1. **How can we scale GNN training and inference to graphs with millions of nodes?**
2. **What is the impact of limited connectivity (cold-start) on GNN performance?**
3. **Can multi-hop neighborhood sampling effectively compensate for cold-start scenarios?**

We focus on node classification as a proxy for recommendation quality, where predicting item categories demonstrates the model's ability to learn meaningful representations.

### 1.3 Objectives

Our specific goals are:

- Reproduce key scalability techniques from LinkedIn's LiGNN framework using public datasets
- Implement GraphSAGE with efficient mini-batch sampling for large graphs
- Quantify the cold-start performance gap and analyze its characteristics
- Establish a baseline for future improvements using attention mechanisms or meta-learning

---

## 2. Related Work

### 2.1 Graph Neural Networks

**Graph Convolutional Networks (GCN)** [Kipf & Welling, 2017] introduced semi-supervised learning on graphs through spectral convolutions. However, GCNs require full-batch processing, making them impractical for large graphs.

**GraphSAGE** [Hamilton et al., 2017] addressed scalability through neighborhood sampling and mini-batch training. Instead of using all neighbors, GraphSAGE samples a fixed number at each layer, enabling training on graphs with billions of edges. The framework supports multiple aggregation functions (mean, LSTM, pooling) and has been successfully deployed in production systems.

**Graph Attention Networks (GAT)** [Veličković et al., 2018] introduced attention mechanisms to weight neighbor contributions dynamically, allowing the model to focus on more relevant connections. This is particularly promising for cold-start scenarios where not all neighbors are equally informative.

### 2.2 Large-Scale GNN Systems

**LiGNN** [LinkedIn, 2020] is a production GNN framework handling billions of nodes and edges. Key innovations include:
- Efficient neighbor sampling strategies
- Distributed training across multiple GPUs
- Online inference with real-time updates
- Handling of cold-start nodes through feature engineering

**PinSage** [Pinterest, 2018] demonstrated GNN-based recommendations at web-scale, using random walks for sampling and importance pooling for aggregation.

### 2.3 Cold-Start Problem

The cold-start problem is well-studied in recommendation systems:

**Traditional approaches** include:
- Content-based filtering using item metadata
- Hybrid methods combining collaborative and content signals
- Transfer learning from related domains

**GNN-specific solutions** include:
- Meta-learning for few-shot node classification [Zhou et al., 2019]
- Attention mechanisms to better utilize limited connections [Veličković et al., 2018]
- Feature augmentation through knowledge graphs [Wang et al., 2019]

Our work builds on GraphSAGE's sampling approach and provides empirical analysis of cold-start performance on a public benchmark, making results reproducible and comparable.

---

## 3. Dataset Description

### 3.1 OGB ogbn-products

We use the **Open Graph Benchmark (OGB) ogbn-products** dataset, which represents an Amazon product co-purchasing network:

| Property | Value |
|----------|-------|
| **Nodes** | 2,449,029 products |
| **Edges** | 61,859,140 co-purchases |
| **Node Features** | 100-dimensional (product metadata) |
| **Classes** | 47 product categories |
| **Task** | Multi-class node classification |
| **Splits** | Pre-defined train/validation/test |

### 3.2 Graph Characteristics

**Degree Distribution**: The graph exhibits a highly skewed degree distribution (Figure 1), typical of real-world networks:
- **Mean degree**: ~50.5
- **Median degree**: 3
- **Max degree**: 27,497
- **Nodes with degree ≤ 2**: 236,337 (9.6% of total)

This power-law distribution creates a natural cold-start scenario: many products have few co-purchases, while popular items are highly connected.

**Connectivity**: The graph contains multiple connected components:
- One large component with >99% of nodes
- Several small isolated components (2-10 nodes)
- This structure reflects real product catalog organization where some items are isolated or in niche categories

**Sparsity**: Despite 61M edges, the graph is very sparse:
- Density: ~1.03 × 10⁻⁵
- Most nodes have local connectivity only
- Sampling strategies are essential for computational efficiency

### 3.3 Cold-Start Definition

We define **cold-start nodes** as those with **degree ≤ 2** (at most 2 connections). This threshold represents products with minimal co-purchase history, mimicking newly added items in a production system.

**Cold-start statistics**:
- Total cold-start nodes: 236,337 (9.6%)
- Validation cold-start: 347 nodes
- Test cold-start: 235,990 nodes

The large test set cold-start population (96.5% of all test nodes) makes this a critical evaluation scenario.

---

## 4. Method

### 4.1 Baseline: MLP

We establish a baseline using a 2-layer Multi-Layer Perceptron (MLP) that processes only node features, ignoring graph structure:

```
MLP:
  Linear(100 → 256) + ReLU
  Linear(256 → 47)
```

This baseline tests whether graph structure provides value beyond content features alone.

### 4.2 GraphSAGE Architecture

**GraphSAGE** (Graph Sample and Aggregate) learns node embeddings through recursive neighborhood aggregation. For node *v* at layer *k*:

1. **Sample**: Select *N_k* neighbors uniformly at random
2. **Aggregate**: Compute neighborhood representation using mean aggregation
3. **Update**: Combine node's previous embedding with aggregated neighborhood

Our implementation uses:

**Architecture**:
- **Input dimension**: 100 (node features)
- **Hidden dimension**: 256
- **Output dimension**: 47 (classes)
- **Layers**: 2 GraphSAGE layers
- **Aggregation**: Mean pooling
- **Activation**: ReLU
- **Total parameters**: 75,567

**Sampling Strategy**:
- **1-hop neighbors**: Sample 10 nodes
- **2-hop neighbors**: Sample 5 nodes per 1-hop neighbor
- Maximum sampled subgraph: 1 + 10 + 10×5 = 61 nodes per target

This sampling dramatically reduces computation from millions to tens of nodes per forward pass.

### 4.3 Training Configuration

**Optimization**:
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch size: 1,024 nodes
- Epochs: 10

**Mini-batch training**:
1. Sample batch of target nodes
2. For each target, sample multi-hop neighborhood
3. Construct induced subgraph with only sampled nodes/edges
4. Forward pass on subgraph
5. Backpropagate gradients for target nodes only

This approach enables training on graphs that don't fit in GPU memory.

### 4.4 Inference and Embeddings

For evaluation, we generate embeddings for all nodes using batched inference:
- Process nodes in batches of 2,048
- For each batch, sample neighborhoods as in training
- Extract embeddings from the penultimate layer (256-dim)
- Store predictions and embeddings for analysis

---

## 5. Experimental Setup

### 5.1 Computational Environment

- **Hardware**: NVIDIA GPU (CUDA-enabled)
- **Software**: Python 3.9, PyTorch 2.0, OGB 1.3.6
- **Code**: Available at https://github.com/Marianna3011/ChoirKids_SWSN

### 5.2 Evaluation Metrics

**Primary metric**: Classification accuracy
- Macro-averaged over all classes
- Reported separately for validation and test sets
- Broken down by cold-start vs. well-connected nodes

**Cold-start analysis**:
- Accuracy on nodes with degree ≤ 2
- Performance gap: difference between all nodes and cold-start nodes
- Statistical significance testing

### 5.3 Evaluation Protocol

We follow OGB's pre-defined splits to ensure fair comparison:
- **Training set**: Used for gradient updates
- **Validation set**: Used for hyperparameter tuning and early stopping
- **Test set**: Final evaluation, reported once

All experiments use the same random seeds for reproducibility.

---

## 6. Results

### 6.1 Overall Performance

| Model | Val Acc (All) | Val Acc (Cold) | Test Acc (All) | Test Acc (Cold) |
|-------|---------------|----------------|----------------|-----------------|
| **MLP Baseline** | 27.29% | — | — | — |
| **GraphSAGE** | **89.27%** | 74.64% | **74.92%** | 65.11% |

**Key findings**:

1. **GraphSAGE dramatically outperforms MLP** (89% vs. 27% validation accuracy), confirming that graph structure contains crucial information beyond node features alone.

2. **Strong generalization**: Test accuracy (74.92%) is reasonably close to validation (89.27%), though the gap suggests some overfitting.

3. **Cold-start remains challenging**: Both validation and test show significant accuracy drops on cold-start nodes.

### 6.2 Cold-Start Performance Gap

| Split | All Nodes | Cold-Start | Gap | Relative Drop |
|-------|-----------|------------|-----|---------------|
| **Validation** | 89.27% | 74.64% | **14.63%** | 16.4% |
| **Test** | 74.92% | 65.11% | **9.82%** | 13.1% |

**Analysis**:

- **Validation gap (14.63%)** is larger than test gap (9.82%), likely due to:
  - Small validation cold-start sample (347 nodes) causing higher variance
  - Different cold-start degree distributions between splits

- **65-75% accuracy on cold-start** shows that 2-hop sampling provides partial compensation:
  - Even with 0-2 direct connections, nodes can access information through 2-hop neighbors
  - Feature similarity between connected products helps

- **Relative performance drop (13-16%)** indicates that while challenging, cold-start nodes are not completely unsolvable with current methods.

### 6.3 Comparison to Baseline

GraphSAGE improvement over MLP:
- **Overall gain**: +62 percentage points (27% → 89%)
- This massive improvement demonstrates the value of neighborhood aggregation

The MLP's poor performance (27%) is only slightly better than random guessing (1/47 = 2.1%), showing that node features alone are insufficient. Graph structure is essential for this task.

### 6.4 Embedding Quality Analysis

We performed t-SNE visualization of learned embeddings (Figure 2, visualization code in notebooks):

**Observations**:
- Well-connected nodes form tight, well-separated clusters by category
- Cold-start nodes show more spread within categories
- Some cold-start nodes cluster well when they connect to informative neighbors
- Others appear between clusters, suggesting uncertain predictions

This visual analysis confirms that cold-start nodes learn less discriminative embeddings due to limited information propagation.

---

## 7. Discussion

### 7.1 Why Does GraphSAGE Work?

GraphSAGE's success stems from:

1. **Multi-hop aggregation**: Nodes access information from neighbors-of-neighbors, expanding the receptive field
2. **Scalable sampling**: Fixed-size neighborhoods enable efficient training on massive graphs
3. **Feature + structure fusion**: Combining node features with graph topology captures complementary signals

### 7.2 Cold-Start Challenge Analysis

The 10-15% performance gap for cold-start nodes reveals:

**Limitations**:
- Limited connections mean fewer information pathways
- Sampling becomes less effective when neighborhoods are small
- High-degree neighbors dominate aggregation in 2-hop scenarios

**Partial successes**:
- 65-75% accuracy shows the method isn't completely failing
- 2-hop sampling helps: even nodes with degree 1-2 can access 10-50 second-order neighbors
- Feature quality (100-dim embeddings) provides baseline signal

### 7.3 Comparison to LiGNN

Our approach captures key LiGNN principles:
- ✅ Neighborhood sampling for scalability
- ✅ Multi-hop aggregation
- ✅ Focus on cold-start scenarios

**Differences from production LiGNN**:
- Simpler architecture (2 layers vs. deeper networks)
- Static graph (vs. real-time updates)
- No feature engineering or side information integration
- Smaller scale (2.5M vs. billions of nodes)

These simplifications allow reproducible research while maintaining core insights.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Shallow architecture**: 2 layers may not capture long-range dependencies
2. **Uniform sampling**: All neighbors weighted equally, ignoring importance
3. **No temporal dynamics**: Static snapshot vs. evolving graph
4. **Limited cold-start strategies**: No feature augmentation or meta-learning

### 8.2 Proposed Improvements

**For Milestone 5 and beyond**:

1. **Graph Attention Networks (GAT)**:
   - Learn attention weights to focus on informative neighbors
   - May help cold-start nodes better utilize their limited connections

2. **Deeper sampling strategies**:
   - Experiment with 3-hop sampling
   - Adaptive sampling based on degree

3. **Feature augmentation**:
   - Integrate product metadata (price, category, brand)
   - Use pre-trained embeddings from product descriptions

4. **Meta-learning approaches**:
   - Train on artificially created cold-start scenarios
   - Learn to adapt quickly with few connections

5. **Hybrid methods**:
   - Combine GNN with content-based recommendations
   - Ensemble predictions for cold-start nodes

---

## 9. Conclusions

This project successfully implemented and evaluated GraphSAGE on a large-scale product recommendation graph, demonstrating:

1. **Scalability**: Mini-batch sampling enables training on 2.45M node graphs with efficient GPU utilization

2. **Strong performance**: 89% validation accuracy shows GNNs can effectively learn from graph structure, vastly outperforming feature-only baselines

3. **Cold-start quantification**: 10-15% performance gap on low-degree nodes confirms this as a significant challenge requiring targeted solutions

4. **Reproducible research**: Using public OGB datasets and open-source code enables comparison and extension by other researchers

**Key takeaway**: While GraphSAGE provides a strong foundation for large-scale recommendations, the cold-start problem remains partially unsolved and motivates continued research into attention mechanisms, feature engineering, and meta-learning approaches.

The project establishes a solid baseline and evaluation protocol for future iterations exploring more sophisticated architectures and cold-start mitigation strategies.

---

## 10. References

1. **LiGNN Framework**  
   LinkedIn Engineering Blog. "LiGNN: Graph Neural Networks at LinkedIn"  
   https://engineering.linkedin.com/blog/2020/gnn-at-linkedin

2. **GraphSAGE**  
   Hamilton, W., Ying, R., & Leskovec, J. (2017).  
   "Inductive Representation Learning on Large Graphs."  
   *NeurIPS 2017.*

3. **Graph Convolutional Networks**  
   Kipf, T. N., & Welling, M. (2017).  
   "Semi-Supervised Classification with Graph Convolutional Networks."  
   *ICLR 2017.*

4. **Graph Attention Networks**  
   Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).  
   "Graph Attention Networks."  
   *ICLR 2018.*

5. **PinSage**  
   Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W. L., & Leskovec, J. (2018).  
   "Graph Convolutional Neural Networks for Web-Scale Recommender Systems."  
   *KDD 2018.*

6. **Open Graph Benchmark**  
   Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., ... & Leskovec, J. (2020).  
   "Open Graph Benchmark: Datasets for Machine Learning on Graphs."  
   *NeurIPS 2020.*

7. **Meta-Learning for Few-Shot Node Classification**  
   Zhou, F., Cao, C., Zhang, K., Trajcevski, G., Zhong, T., & Geng, J. (2019).  
   "Meta-GNN: On Few-shot Node Classification in Graph Meta-learning."  
   *CIKM 2019.*

8. **Knowledge Graph Enhanced Recommendations**  
   Wang, H., Zhang, F., Zhao, M., Li, W., Xie, X., & Guo, M. (2019).  
   "Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation."  
   *WWW 2019.*

---

## Appendix A: Model Architecture Details

### GraphSAGE Layer Implementation

```python
class SAGEConvLayer(torch.nn.Module):
    """GraphSAGE convolution layer with mean aggregation"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_neighbor = torch.nn.Linear(in_channels, out_channels)
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Aggregate neighbors (mean aggregation)
        row, col = edge_index
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, row, x[col])
        
        degree = torch.bincount(row, minlength=x.size(0))
        neighbor_mean = neighbor_sum / degree.clamp(min=1).unsqueeze(1)
        
        # Combine self and neighbor representations
        return self.lin_self(x) + self.lin_neighbor(neighbor_mean)
```

### Full Model

```python
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConvLayer(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConvLayer(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConvLayer(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x
```

---

## Appendix B: Hyperparameter Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Hidden dimensions | 256 | Balance between capacity and efficiency |
| Number of layers | 2 | Sufficient for 2-hop aggregation |
| Learning rate | 0.001 | Standard Adam default |
| Batch size | 1024 | Maximizes GPU utilization |
| 1-hop samples | 10 | Balances information and computation |
| 2-hop samples | 5 | Limits subgraph to ~61 nodes |
| Epochs | 10 | Sufficient for convergence on this task |
| Aggregation | Mean | Simple and effective for product similarity |

---

## Appendix C: Reproducibility Checklist

✅ **Code**: Public repository with clear structure  
✅ **Data**: Using standard OGB benchmark  
✅ **Seeds**: Fixed random seeds documented  
✅ **Environment**: Requirements.txt provided  
✅ **Instructions**: README with setup and execution steps  
✅ **Results**: All metrics and figures reproducible from notebooks  

Repository: https://github.com/Marianna3011/ChoirKids_SWSN
