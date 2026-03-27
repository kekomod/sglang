# TurboQuant Paper Summary

**Paper:** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
**arXiv:** [2504.19874](https://arxiv.org/abs/2504.19874) (April 28, 2025)
**Venue:** ICLR 2026
**Authors:** Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)

---

## 1. Problem Definition

Design a quantization map Q : R^d -> {0,1}^B that transforms d-dimensional vectors to B-bit binary strings, with an inverse (dequantization) map Q^{-1} : {0,1}^B -> R^d.

Setting B = b * d gives b bits per coordinate on average.

### Distortion Metrics

**MSE distortion:**
```
D_mse := E_Q [||x - Q^{-1}(Q(x))||^2_2]
```

**Inner-product distortion:**
```
D_prod := E_Q [|<y,x> - <y, Q^{-1}(Q(x))>|^2]
```

**Unbiasedness requirement** (for inner products):
```
E_Q [<y, Q^{-1}(Q(x))>] = <y, x>
```

The quantizer Q may be randomized. The expectations are over Q's randomness.

---

## 2. Mathematical Foundations

### 2.1 Coordinate Distribution After Random Rotation (Lemma 1)

For any vector x on the unit sphere S^{d-1}, after applying a random orthogonal rotation Pi, each coordinate follows a Beta distribution:

```
x_j ~ f_X(x) := Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
```

for x in [-1, 1].

**In high dimensions**, this converges to the normal distribution N(0, 1/d).

**Implementation note:** This distribution is the foundation for codebook design. The codebook centroids are optimized for this specific distribution, NOT for a uniform or generic normal distribution.

### 2.2 Shannon Lower Bound on Distortion (Lemma 2-3)

For any compression algorithm Q with bit-width b operating on unit sphere S^{d-1}:

```
D_mse(Q) >= 1/4^b
D_prod(Q) >= (||y||^2 / d) * (1/4^b)
```

These are information-theoretic limits — no algorithm can do better.

### 2.3 QJL: 1-bit Inner Product Quantizer (Definition 1)

The Quantized Johnson-Lindenstrauss (QJL) map:

```
Q_qjl(x) := sign(S * x)    for x in R^d
```

where S in R^{d x d} has i.i.d. N(0,1) entries.

**Dequantization:**
```
Q_qjl^{-1}(z) := (sqrt(pi/2) / d) * S^T * z    for z in {-1, +1}^d
```

**Properties (Lemma 4):**
- **Unbiased:** E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x>
- **Variance bound:** Var(<y, Q_qjl^{-1}(Q_qjl(x))>) <= (pi / (2d)) * ||y||^2_2

**Implementation note:** The scale factor sqrt(pi/2)/d is critical for unbiasedness. The matrix S is a dense random Gaussian matrix (NOT orthogonalized). One matrix per layer, shared across heads.

---

## 3. Algorithms

### 3.1 Algorithm 1: TurboQuant_mse (MSE-Optimal)

**Purpose:** Minimize reconstruction MSE. Used for **value** quantization.

**Setup (one-time):**
1. Generate a random rotation matrix Pi in R^{d x d} via QR decomposition on a matrix with i.i.d. N(0,1) entries. Apply sign correction: `Pi *= sign(diag(R))` for Haar-uniform distribution.
2. Construct codebook: find centroids c_1, ..., c_{2^b} in [-1,1] that minimize the 1D k-means cost for the Beta distribution f_X (Eq. 4).

**Procedure QUANT_mse(x):**
1. `y = Pi * x` (rotate)
2. For each j in [d]: `idx_j = argmin_k |y_j - c_k|` (nearest centroid, b-bit index)
3. Output: idx (vector of b-bit integers)

**Procedure DEQUANT_mse(idx):**
1. For each j: `y_tilde_j = c_{idx_j}` (lookup centroid)
2. `x_tilde = Pi^T * y_tilde` (inverse rotate)
3. Output: x_tilde

**For non-unit vectors:** Store the L2 norm separately in floating point. Normalize before quantization, rescale after dequantization.

### 3.2 Codebook Computation (Eq. 4)

The codebook is found by solving the continuous 1D k-means problem:

```
C(f_X, b) := min_{-1 <= c_1 <= ... <= c_{2^b} <= 1}
             sum_{i=1}^{2^b} integral_{(c_{i-1}+c_i)/2}^{(c_i+c_{i+1})/2} |x - c_i|^2 * f_X(x) dx
```

**Solved via the Max-Lloyd iterative algorithm:**
1. Initialize centroids at CDF quantiles: `c_i = CDF^{-1}((i + 0.5) / 2^b)`
2. Repeat until convergence:
   a. Compute Voronoi boundaries: `b_i = (c_i + c_{i+1}) / 2`
   b. Update centroids: `c_i = integral_{b_{i-1}}^{b_i} x * f_X(x) dx / integral_{b_{i-1}}^{b_i} f_X(x) dx`

**Analytical codebook values for high-dimensional limit (f_X -> N(0, 1/d)):**
- b=1: {+/- sqrt(2/pi) / sqrt(d)} = {+/- 0.7979 / sqrt(d)}
- b=2: {+/- 0.453 / sqrt(d), +/- 1.51 / sqrt(d)}
- b=3, b=4: computed numerically

**Implementation note:** Compute codebooks once at initialization. Cache by (dimension, bits). Use numerical integration on a fine grid (e.g., 32768 points) for the Beta PDF.

### 3.3 Algorithm 2: TurboQuant_prod (Inner-Product-Optimal)

**Purpose:** Minimize inner-product distortion with unbiased estimation. Used for **key** quantization.

**Why not just use TurboQuant_mse for keys?** Because MSE-optimal quantizers are biased for inner products. At b=1, the bias is a multiplicative factor of 2/pi. This bias diminishes with increasing b but is significant at low bit-widths.

**Setup (one-time):**
1. Instantiate a TurboQuant_mse with bit-width **b - 1** (one fewer bit)
2. Generate a random projection matrix S in R^{d x d} with S_{i,j} ~ N(0,1)

**Procedure QUANT_prod(x):**
1. `idx = Quant_mse(x)` (using b-1 bits)
2. `r = x - DeQuant_mse(idx)` (compute residual)
3. `qjl = sign(S * r)` (1-bit QJL on residual)
4. Output: (idx, qjl, ||r||_2)

**Procedure DEQUANT_prod(idx, qjl, gamma):**
1. `x_tilde_mse = DeQuant_mse(idx)`
2. `x_tilde_qjl = (sqrt(pi/2) / d) * gamma * S^T * qjl`
3. Output: `x_tilde_mse + x_tilde_qjl`

**For computing inner products directly (fused attention):**
```
<y, x_tilde> = <y, x_tilde_mse> + gamma * sqrt(pi/2)/d * <y, S^T * qjl>
             = <y, x_tilde_mse> + gamma * sqrt(pi/2)/d * <S*y, qjl>
```

This means the inner product can be computed WITHOUT full dequantization:
- Precompute `y_rot = Pi * y` for the MSE component: `<y, x_tilde_mse> = <y_rot, codebook[idx]>`
- Precompute `y_proj = S * y` for the QJL component: `<y_proj, qjl>` is a dot product with signs

**Implementation note:** This fused computation is the key to performance. The query is projected once, then inner products with ALL cached keys are computed from packed bit representations.

### 3.4 Bit Budget

For TurboQuant_prod at b total bits per dimension:
- (b-1) bits for the MSE codebook indices
- 1 bit for the QJL sign
- Plus: 1 float (16-bit) for vector norm, 1 float for residual norm → negligible overhead for d >> 1

Example at b=3 (3 bits per dim):
- 2-bit MSE indices: 4 centroids per dimension
- 1-bit QJL signs
- Storage per token per head: 3 * 256 / 8 = 96 bytes (vs 512 bytes for FP16) = 5.3x compression

---

## 4. Distortion Bounds

### 4.1 MSE Distortion (Theorem 1)

For any bit-width b >= 1 and any vector x on S^{d-1}:

```
D_mse <= (sqrt(3) * pi / 2) * (1 / 4^b)
```

Refined values for small b:
| b | D_mse upper bound |
|---|---|
| 1 | 0.36 |
| 2 | 0.117 |
| 3 | 0.03 |
| 4 | 0.009 |

### 4.2 Inner-Product Distortion (Theorem 2)

For TurboQuant_prod with b total bits:

- **Unbiased:** E[<y, x_tilde>] = <y, x>
- **Distortion:** D_prod <= (sqrt(3) * pi^2 * ||y||^2 / d) * (1 / 4^b)

Refined values for small b (normalized by ||y||^2/d):
| b | D_prod * d / ||y||^2 |
|---|---|
| 1 | 1.57 |
| 2 | 0.56 |
| 3 | 0.18 |
| 4 | 0.047 |

### 4.3 Lower Bounds (Theorem 3)

For ANY randomized quantization algorithm with bit-width b:

```
D_mse >= 1 / 4^b
D_prod >= (||y||^2 / d) * (1 / 4^b)
```

**Gap to optimality:** TurboQuant is within a factor of sqrt(3)*pi/2 ~ 2.7 of the information-theoretic lower bound. At b=1, the gap is only ~1.45.

---

## 5. Outlier Channel Handling (Section 4.3)

For non-integer bit-widths, the paper uses a **split-channel** approach:

1. Divide the d dimensions into two groups: outlier channels and regular channels
2. Apply TurboQuant at different bit-widths to each group
3. The effective bit-width is the weighted average

**Example (2.5 bits, d=128):**
- 32 outlier channels at 3 bits
- 96 regular channels at 2 bits
- Effective: (32 * 3 + 96 * 2) / 128 = 2.5 bits

**Example (3.5 bits, d=128):**
- Different ratio of channels at 3 vs 4 bits

**Outlier selection:** The paper identifies outlier channels based on magnitude statistics. The MLX reference implementation uses mean absolute activation across a calibration batch.

**Implementation note:** Each group gets its own codebook and rotation matrix (or shared rotation, split codebook). The two groups are quantized and dequantized independently, then concatenated.

---

## 6. Experimental Results

### 6.1 Empirical Validation (Section 4.1)

Tested on DBpedia Entities dataset (1536-dimensional OpenAI-3 embeddings):
- TurboQuant_prod is unbiased for inner products at all bit-widths
- TurboQuant_mse has bias that decreases with increasing b
- Both match theoretical distortion bounds closely

### 6.2 Needle-in-a-Haystack (Section 4.2)

Model: Llama-3.1-8B-Instruct, sequences 4K-104K tokens, 25% memory compression ratio.

| Method | Score |
|---|---|
| Full Precision | 0.997 |
| **TurboQuant** | **0.997** |
| PolarQuant | 0.995 |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

### 6.3 LongBench-E (Section 4.3, Table 1)

Model: Llama-3.1-8B-Instruct

| Method | KV Bits | SingleQA | MultiQA | Summ. | Few-shot | Synth. | Code | **Avg** |
|---|---|---|---|---|---|---|---|---|
| Full Cache | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | **50.06** |
| KIVI | 3 | 43.38 | 37.99 | 27.16 | 68.38 | 59.50 | 44.68 | **48.50** |
| KIVI | 5 | 45.04 | 45.70 | 26.47 | 68.57 | 59.55 | 46.41 | **50.16** |
| PolarQuant | 3.9 | 45.18 | 44.48 | 26.23 | 68.25 | 60.07 | 45.24 | **49.78** |
| TurboQuant | 2.5 | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | **49.44** |
| TurboQuant | 3.5 | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | **50.06** |

Also tested on Ministral-7B-Instruct:
| Full Cache | 16 | 47.53 | 49.06 | 26.09 | 66.83 | 53.50 | 47.90 | 49.89 |
| TurboQuant | 2.5 | 48.38 | 49.22 | 24.91 | 66.69 | 53.17 | 46.83 | 49.62 |

### 6.4 Quantization Speed (Table 2)

4-bit quantization time in seconds (on A100 GPU):

| Method | d=200 | d=1536 | d=3072 |
|---|---|---|---|
| Product Quantization | 37.04 | 239.75 | 494.42 |
| RabitQ | 597.25 | 2267.59 | 3957.19 |
| **TurboQuant** | **0.0007** | **0.0013** | **0.0021** |

TurboQuant is 170,000x to 1,900,000x faster than alternatives for nearest-neighbor search indexing.

### Addendum: Google Research Blog — Attention Logit Speedup

*Note: This section references the [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), not the arXiv paper itself.*

From the blog (H100 GPU): up to **8x speedup** for attention logit computation in 4-bit mode vs FP32 unquantized keys. This is specifically about computing Q·K scores directly from quantized data (codebook gather + partial dot) vs FP32 dot products — NOT end-to-end inference speedup.

---

## 7. Key Takeaways for Implementation

1. **Keys need TurboQuant_prod** (inner-product-optimized, unbiased) because attention scores are inner products
2. **Values need TurboQuant_mse** (MSE-optimized) because values are weighted-summed after softmax
3. **Codebooks must be computed for the exact Beta distribution**, not approximated as uniform or normal
4. **The rotation matrix must be Haar-uniform orthogonal** (QR + sign correction), not just any orthogonal matrix
5. **The QJL scale factor sqrt(pi/2)/d is critical** — getting it wrong breaks unbiasedness
6. **Inner products can be computed in compressed domain** without full dequantization — this is the key to performance
7. **The method is data-oblivious** — no calibration data needed (unlike GPTQ, AWQ)
8. **Outlier channel handling** enables fractional bit-widths for fine-grained quality-compression tradeoff
9. **All experiments on A100** — compatible with our target hardware
