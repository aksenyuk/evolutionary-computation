# Report

Team members:

- Sofya Aksenyuk, 150284
- Uladzimir Ivashka, 150281

## Problem Description

Given a set of nodes, each characterized by their (x, y) coordinates in a plane and an associated cost, the challenge is to select exactly 50% of these nodes and form a Hamiltonian cycle. 

The goal is to minimize the sum of the total length of the path plus the total cost of the selected nodes. 

Distances between nodes are computed as Euclidean distances and rounded to the nearest integer. 

## Source code

Link: [Source Code](https://github.com/aksenyuk/evolutionary-computation/blob/main/global-convexity/global_convexity.ipynb)

## Plots

See plots: [Plots](https://github.com/aksenyuk/evolutionary-computation/tree/main/global-convexity/plots/)

<div style="page-break-after: always"></div>

<img src="plots/TSPA.png" height="300"/><img src="plots/TSPB.png" height="300"/>

<img src="plots/TSPC.png" height="300"/><img src="plots/TSPD.png" height="300"/>

<div style="page-break-after: always"></div>

## Conclusions

### Consistent Trends

Plots for all TSP instances show similar shapes and tendencies, indicating a general pattern applicable across various problems.

### Cost-Similarity Correlation

There is a notable trend where lower-cost solutions exhibit higher similarities, suggesting that solutions closer to the optimal tend to be more alike, especially in their node configurations.

### Edge vs. Node Similarity

Edge similarity to the best solution shows less dense clustering and is generally lower than node similarity, implying more variation in edge selections among high-quality solutions.

### Coefficient Correlation Insights

Correlation coefficients are mostly above 0.5, showing a moderate to strong link between cost and similarity.

Average edge similarity to local optima consistently shows higher correlation (around 0.6-0.7) compared to other aspects, highlighting the importance of edge selection in solution quality.