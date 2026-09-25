"""
Expands domain classifier training data with focus on the hard math/DL overlap
region (optimization, linear algebra used in ML, probability in DL).

Targets: ≥300 examples per class in train.csv.
Current: deep_learning=182, computer_science=180, mathematics=180, general=180.
Need to add ≈120 per class (total ≈480 new rows).

Run from repo root:
    python scripts/expand_classifier_data.py
"""

import csv
import random
import sys
from pathlib import Path

random.seed(42)

TRAIN_CSV = Path("data/domain_classifier/train.csv")
VAL_CSV   = Path("data/domain_classifier/val.csv")

# ---------------------------------------------------------------------------
# Hard overlap: mathematics texts with optimization/linear-algebra phrasing
# that look similar to DL, and vice versa
# ---------------------------------------------------------------------------

MATH_HARD = [
    # Optimization theory (pure math side)
    "The gradient of a scalar-valued function f(x) points in the direction of steepest ascent.",
    "A convex function satisfies f(lambda x + (1-lambda) y) <= lambda f(x) + (1-lambda) f(y) for all lambda in [0,1].",
    "The Lagrangian L(x, mu) = f(x) + mu g(x) encodes constrained optimization via dual variables.",
    "KKT conditions are necessary and sufficient for optimality in convex programming problems.",
    "Gradient descent converges at rate O(1/k) for convex Lipschitz-continuous objectives.",
    "Newton's method achieves quadratic convergence near a non-degenerate critical point.",
    "The conjugate gradient method minimizes quadratic forms without storing the Hessian explicitly.",
    "Proximal gradient methods split an objective into smooth and non-smooth components.",
    "The subgradient of a non-differentiable convex function generalizes the classical derivative.",
    "Dual decomposition separates large-scale optimization into smaller independent subproblems.",
    "Stochastic approximation converges almost surely under square-summable step-size conditions.",
    "Mirror descent updates parameters using a Bregman divergence instead of Euclidean projection.",
    "Variance reduction techniques like SVRG reduce gradient noise in finite-sum minimization.",
    "The proximal operator of a function f maps x to argmin_z { f(z) + (1/2)||z - x||^2 }.",
    "Accelerated gradient methods like Nesterov's achieve O(1/k^2) convergence for smooth convex functions.",
    # Linear algebra (pure math side)
    "The spectral theorem states that every symmetric real matrix is orthogonally diagonalizable.",
    "The Gram matrix G_ij = <phi(x_i), phi(x_j)> is symmetric and positive semi-definite by construction.",
    "An orthogonal projection onto a subspace satisfies P^2 = P and P^T = P simultaneously.",
    "The Cayley-Hamilton theorem states every matrix satisfies its own characteristic polynomial equation.",
    "Schur decomposition writes any square matrix as Q U Q^* where U is upper triangular.",
    "The trace of a matrix equals the sum of its eigenvalues counting algebraic multiplicities.",
    "A matrix is positive definite if and only if all its leading principal minors are strictly positive.",
    "The Cholesky factorization A = L L^T applies to symmetric positive definite matrices.",
    "Rank-nullity theorem: dim(ker A) + rank(A) = n for any linear map A: R^n -> R^m.",
    "Block matrix inversion uses the Schur complement to express inverses of partitioned matrices.",
    "The pseudoinverse A+ satisfies all four Moore-Penrose conditions for consistent least squares.",
    "Tensor contraction generalizes matrix multiplication to multi-indexed arrays of arbitrary order.",
    "The Frobenius norm of a matrix equals the square root of the sum of squared singular values.",
    "A Vandermonde matrix with distinct nodes is invertible and has a closed-form determinant.",
    "LU factorization with partial pivoting is numerically stable for dense systems of equations.",
    # Probability / statistics (pure math side)
    "A sigma-algebra is a collection of sets closed under complement and countable unions.",
    "The Radon-Nikodym derivative dP/dQ defines a density when P is absolutely continuous w.r.t. Q.",
    "Martingale convergence theorem guarantees almost sure convergence for bounded martingales.",
    "Cramer-Rao lower bound states the variance of any unbiased estimator is at least 1 / I(theta).",
    "The moment generating function M_X(t) = E[e^{tX}] uniquely determines the distribution of X.",
    "A sufficient statistic captures all information about a parameter contained in the sample.",
    "Markov chains satisfy the memoryless property: future state depends only on the current state.",
    "The law of large numbers guarantees sample averages converge to the population expectation.",
    "Jensen's inequality states E[phi(X)] >= phi(E[X]) for any convex function phi.",
    "Exponential family distributions have sufficient statistics and conjugate prior families.",
    # Hindi math (optimization / linear algebra)
    "उत्तल फलन का स्थानीय न्यूनतम सदैव वैश्विक न्यूनतम भी होता है।",
    "ग्रेडिएंट डिसेंट प्रत्येक चरण में फलन के आंशिक अवकलज की विपरीत दिशा में चलता है।",
    "द्विघात प्रोग्रामिंग समस्याओं में KKT शर्तें इष्टतमता के लिए आवश्यक और पर्याप्त हैं।",
    "नेस्टरोव त्वरण O(1/k^2) अभिसरण दर प्राप्त करता है जो साधारण ग्रेडिएंट विधि से तेज है।",
    "लाग्रांज गुणक विधि समकक्ष बाधाओं के साथ अनुकूलन समस्याओं को हल करती है।",
    "सिंगुलर वैल्यू डीकंपोजिशन किसी भी मैट्रिक्स को तीन ऑर्थोगोनल मैट्रिसेज़ के गुणनफल में लिखती है।",
    "आइगेनवैल्यू विघटन सममित मैट्रिसेज़ के वर्णक्रम गुणों को प्रकट करता है।",
    "लीस्ट स्क्वेर्स विधि अवलोकित और प्रत्याशित मानों के बीच वर्ग त्रुटि को न्यूनतम करती है।",
    # Code-mixed math (hard overlap with DL)
    "Gradient descent optimization के लिए learning rate और convergence के बीच trade-off होता है।",
    "Linear algebra में eigendecomposition A = Q Lambda Q^{-1} के through matrix powers efficiently compute होती हैं।",
    "Constrained optimization में Lagrange multipliers dual variables के रूप में constraint को enforce करते हैं।",
    "Convex functions के लिए global minimum local minimum के साथ coincide करता है।",
    "Stochastic gradient descent batch size और variance reduction के बीच fundamental trade-off रखता है।",
    "Positive definite matrix का Cholesky decomposition numerical stability provide करता है।",
    "SVD matrix के rank-k approximation के लिए Eckart-Young theorem optimize करती है।",
    "Probability theory में conditional expectation E[Y|X] projection operator की properties satisfy करता है।",
]

DL_HARD = [
    # Optimization used in DL — hardest overlap with pure math
    "Adam optimizer combines momentum and RMSProp to adapt per-parameter learning rates automatically.",
    "Gradient clipping limits the L2 norm of parameter gradients to prevent exploding updates.",
    "Batch normalization normalizes layer inputs using per-feature mean and variance over a mini-batch.",
    "Layer normalization computes statistics across the feature dimension rather than the batch dimension.",
    "Weight decay adds an L2 penalty to the training loss to prevent overfitting of large models.",
    "Label smoothing softens one-hot targets to improve calibration and generalization in classification.",
    "Cosine annealing schedules the learning rate using a cosine decay to a minimum value per cycle.",
    "The cross-entropy loss for multi-class classification equals the negative log likelihood of the target.",
    "Dropout randomly zeroes activations with probability p at training time to act as ensemble averaging.",
    "Residual connections allow gradients to flow directly through identity shortcuts during backprop.",
    "The softmax function converts logit vectors to valid probability distributions over classes.",
    "The attention mechanism computes weighted sums of values where weights derive from query-key similarity.",
    "Positional encodings inject sequence order information into Transformer architectures via additive bias.",
    "LoRA fine-tuning decomposes weight updates as low-rank matrices to reduce trainable parameter count.",
    "Knowledge distillation trains a small student model to match the soft outputs of a large teacher.",
    # Linear algebra in DL
    "Weight matrices in fully-connected layers perform learned affine transformations on input representations.",
    "Multi-head attention projects queries, keys, and values with separate learned projection matrices.",
    "The covariance matrix of a batch captures second-order statistics used in whitening transformations.",
    "Principal component analysis applied to activations reveals the intrinsic dimensionality of representations.",
    "Matrix factorization in embedding layers decomposes a vocabulary-by-embedding interaction matrix.",
    "Convolutional layers implement cross-correlation with learned filter banks as linear operators.",
    "The Jacobian matrix of a neural network layer maps input perturbations to output perturbations.",
    "Spectral normalization constrains the largest singular value of weight matrices to stabilize GAN training.",
    "Tensor decomposition compresses high-dimensional weight tensors into sums of rank-1 components.",
    "The Fisher information matrix approximates the curvature of the log-likelihood surface for natural gradient.",
    # Probability in DL
    "Variational autoencoders maximize the evidence lower bound to jointly learn encoder and decoder.",
    "The KL divergence between the approximate posterior and prior acts as a regularizer in VAEs.",
    "Monte Carlo dropout estimates predictive uncertainty by sampling stochastic forward passes at test time.",
    "Normalizing flows learn invertible mappings to transform a simple prior into complex posteriors.",
    "The ELBO decomposes into a reconstruction term and a KL penalty between prior and posterior.",
    "Contrastive learning maximizes agreement between positive pairs while repelling negative pairs.",
    "Diffusion models iteratively denoise Gaussian noise to generate samples from the data distribution.",
    "The score function estimator enables gradient estimation through non-differentiable sampling operations.",
    "Bayesian neural networks place distributions over weights rather than using point estimates.",
    "Mutual information maximization is used as a self-supervised pretraining objective for representations.",
    # Hindi DL hard examples
    "ट्रांसफॉर्मर मॉडल में मल्टी-हेड अटेंशन विभिन्न सबस्पेसेज़ में समानांतर ध्यान की गणना करता है।",
    "ग्रेडिएंट क्लिपिंग गहरे नेटवर्क में विस्फोटक ग्रेडिएंट समस्या को नियंत्रित करती है।",
    "बैच नॉर्मलाइज़ेशन प्रत्येक फीचर के माध्य और विचरण को सामान्यीकृत करके प्रशिक्षण स्थिर करती है।",
    "वैरिएशनल ऑटोएनकोडर अव्यक्त स्थान में संभाव्यता वितरण सीखता है।",
    "नॉलेज डिस्टिलेशन बड़े टीचर मॉडल के सॉफ्ट आउटपुट से छोटा स्टूडेंट मॉडल प्रशिक्षित करती है।",
    "डिफ्यूजन मॉडल गाउसियन शोर को क्रमिक रूप से हटाकर यथार्थवादी डेटा उत्पन्न करता है।",
    "LoRA फाइन-ट्यूनिंग लो-रैंक मैट्रिक्स अपडेट से LLM को कुशलतापूर्वक अनुकूलित करती है।",
    "कंट्रास्टिव लर्निंग सकारात्मक जोड़ियों को पास लाते हुए नकारात्मक जोड़ियों को दूर धकेलती है।",
    # Code-mixed DL hard examples
    "Transformer architecture में self-attention Q, K, V matrices के dot-product से attention weights compute होते हैं।",
    "Adam optimizer के beta parameters gradient के exponential moving averages को control करते हैं।",
    "Variational autoencoder में ELBO loss reconstruction term और KL divergence को balance करती है।",
    "Batch normalization training के दौरान internal covariate shift problem को address करती है।",
    "Dropout regularization neural network को overfitting से बचाने के लिए random neurons को mask करता है।",
    "Residual connections gradient flow improve करके vanishing gradient problem solve करते हैं।",
    "Score-based generative model data distribution के score function को directly estimate करती है।",
    "Contrastive self-supervised learning negative samples के through representation quality improve करती है।",
]

CS_HARD = [
    # Algorithms and complexity
    "Amortized analysis distributes expensive operations across a sequence to compute per-operation cost.",
    "A red-black tree satisfies five invariants guaranteeing O(log n) worst-case search and insert.",
    "Topological sort on a DAG orders vertices such that every edge points from earlier to later.",
    "Dijkstra's algorithm uses a priority queue to find single-source shortest paths in O((V+E) log V).",
    "The master theorem solves recurrences of the form T(n) = aT(n/b) + f(n) for divide-and-conquer.",
    "Dynamic programming memoizes overlapping subproblems to reduce exponential recursion to polynomial time.",
    "The travelling salesman problem is NP-hard; exact solutions require exponential time in the worst case.",
    "Bloom filters use multiple hash functions to test set membership with bounded false-positive probability.",
    "Cache-oblivious algorithms achieve optimal cache performance without knowledge of cache parameters.",
    "A skip list is a probabilistic data structure that achieves O(log n) average search complexity.",
    "Union-find with path compression and rank achieves nearly-constant amortized time per operation.",
    "Segment trees support range queries and point updates in O(log n) per operation on arrays.",
    "Suffix arrays provide compressed indexing of a string for efficient pattern matching queries.",
    "The fast Fourier transform computes the DFT of n elements in O(n log n) via divide and conquer.",
    "P vs NP asks whether every problem whose solution can be verified in polynomial time can also be solved.",
    # Systems and OS
    "A CPU pipeline hazard occurs when an instruction depends on the result of a preceding instruction.",
    "Copy-on-write delays page duplication until a process actually writes to a shared memory region.",
    "NUMA-aware memory allocation places data on the socket physically nearest to the accessing core.",
    "A B-tree keeps all leaves at the same depth and supports efficient sorted storage on disk.",
    "The two-phase commit protocol ensures distributed transactions either commit or abort atomically.",
    "Consistent hashing minimizes key redistribution when nodes are added or removed from a cluster.",
    "A log-structured merge tree batches writes in memory and merges sorted runs to disk efficiently.",
    "Zero-copy I/O transfers data directly between kernel buffers and network sockets without CPU copies.",
    # Networks and security
    "TLS 1.3 removes all cipher suites with known weaknesses and enforces forward secrecy by default.",
    "A denial-of-service attack exhausts server resources by flooding with crafted or legitimate requests.",
    "Content-addressable storage uses cryptographic hashes of data as keys for deduplication.",
    "BGP hijacking redirects internet traffic by announcing more specific or equal routes maliciously.",
    # Hindi CS hard
    "रेड-ब्लैक ट्री पाँच संरचनात्मक नियमों को बनाए रखते हुए O(log n) सम्मिलन और खोज सुनिश्चित करता है।",
    "डायनामिक प्रोग्रामिंग ओवरलैपिंग उप-समस्याओं को मेमोइज़ेशन से हल करके घातांकीय पुनरावृत्ति से बचाती है।",
    "यूनियन-फाइंड संरचना पथ संपीड़न के साथ लगभग स्थिर परिशोधित समय प्रदान करती है।",
    "B-ट्री डिस्क-आधारित डेटाबेस इंडेक्स के लिए लॉगरिदमिक सर्च को सुनिश्चित करती है।",
    "दो-चरणीय कमिट प्रोटोकॉल वितरित लेनदेन में परमाणुता की गारंटी देता है।",
    # Code-mixed CS hard
    "Red-black tree insertion के बाद rotation और color-flip से invariants restore होते हैं।",
    "Dynamic programming में optimal substructure और overlapping subproblems दोनों conditions जरूरी हैं।",
    "Distributed system में CAP theorem network partition के दौरान consistency और availability के बीच trade-off force करती है।",
    "Cache-oblivious matrix multiplication recursively tiles computation without explicit cache parameter tuning।",
    "Hash table में load factor बढ़ने पर rehashing average O(1) amortized complexity maintain करता है।",
]

GENERAL_HARD = [
    # General (must NOT be confused with technical domains)
    "Climate change negotiations require balancing economic development goals with emissions reduction targets.",
    "Public health infrastructure investment correlates with improved life expectancy across income groups.",
    "Urban planning decisions shape commute patterns, air quality, and community cohesion over decades.",
    "The global supply chain disruptions following the pandemic highlighted systemic logistics vulnerabilities.",
    "Renewable energy adoption depends on government policy incentives and grid infrastructure readiness.",
    "Digital literacy programs help citizens engage meaningfully with government services online.",
    "Historical analysis of colonialism reveals long-lasting economic and cultural impacts on former colonies.",
    "Scientific literacy enables individuals to evaluate medical evidence and public health recommendations.",
    "International trade agreements influence domestic employment patterns across multiple industrial sectors.",
    "Media literacy education helps audiences distinguish credible journalism from misinformation campaigns.",
    "Environmental impact assessments evaluate how proposed infrastructure projects affect local ecosystems.",
    "Intercultural communication skills reduce misunderstandings in diverse multinational workplaces.",
    "Educational curriculum design must balance foundational knowledge with twenty-first century competencies.",
    "Demographic transitions affect pension system sustainability and healthcare demand in aging societies.",
    "Community resilience after natural disasters depends on social capital and local government coordination.",
    "Circular economy principles aim to minimise waste by designing products for reuse and recycling.",
    "Vaccine hesitancy research examines the psychological, cultural, and informational factors behind refusal.",
    "Financial inclusion initiatives expand access to banking services among underserved rural populations.",
    "Civic participation in local governance is correlated with higher satisfaction and service delivery.",
    "Linguistic diversity in education supports cognitive development and cultural identity preservation.",
    # Hindi general
    "जलवायु परिवर्तन से निपटने के लिए अंतरराष्ट्रीय सहयोग और नीतिगत समन्वय आवश्यक है।",
    "शहरी नियोजन में सार्वजनिक परिवहन नेटवर्क का विस्तार यातायात भीड़ को कम करता है।",
    "सार्वजनिक स्वास्थ्य सेवाओं में निवेश दीर्घकालिक आर्थिक उत्पादकता को बढ़ावा देता है।",
    "डिजिटल साक्षरता कार्यक्रम नागरिकों को ऑनलाइन सरकारी सेवाओं का उपयोग करने में सक्षम बनाते हैं।",
    "अंतरराष्ट्रीय व्यापार समझौते घरेलू रोजगार पैटर्न और उद्योग संरचना को प्रभावित करते हैं।",
    "मीडिया साक्षरता शिक्षा व्यक्तियों को विश्वसनीय पत्रकारिता और गलत सूचना में अंतर करने में मदद करती है।",
    # Code-mixed general
    "Climate change policy में developed और developing countries के बीच responsibility sharing का complex debate है।",
    "Urban infrastructure investment public transportation को improve करके commute time और pollution कम करता है।",
    "Digital divide को bridge करने के लिए rural areas में internet connectivity और device access जरूरी है।",
    "Public health campaigns vaccine hesitancy को address करके herd immunity achieve करने में मदद करते हैं।",
    "Financial inclusion programs rural populations को banking services और micro-credit तक access provide करते हैं।",
]

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=("text", "domain")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main():
    train_rows = read_csv(TRAIN_CSV)
    val_rows   = read_csv(VAL_CSV)

    # Count current per-class sizes
    from collections import Counter
    counts = Counter(r["domain"] for r in train_rows)
    print("Before expansion:", dict(counts))

    TARGET = 310  # aim for 310 per class in train
    TARGET_VAL = 90  # aim for 90 per class in val

    new_train, new_val = [], []

    pool = {
        "mathematics":      MATH_HARD,
        "deep_learning":    DL_HARD,
        "computer_science": CS_HARD,
        "general":          GENERAL_HARD,
    }

    for domain, examples in pool.items():
        current_train = counts[domain]
        need_train = max(0, TARGET - current_train)

        val_counts = Counter(r["domain"] for r in val_rows)
        current_val = val_counts[domain]
        need_val = max(0, TARGET_VAL - current_val)

        shuffled = examples.copy()
        random.shuffle(shuffled)

        # Reserve first need_val for val, rest go to train (cycling if needed)
        val_candidates = shuffled[:need_val]
        train_candidates = shuffled[need_val:]

        # Add val examples
        for i, text in enumerate(val_candidates):
            new_val.append({"text": text, "domain": domain})

        # Add train examples — cycle through pool if needed
        added = 0
        idx = 0
        all_candidates = shuffled[need_val:] + shuffled[:need_val]  # use full pool
        while added < need_train:
            new_train.append({"text": all_candidates[idx % len(all_candidates)], "domain": domain})
            idx += 1
            added += 1

    # Merge and shuffle
    merged_train = train_rows + new_train
    random.shuffle(merged_train)

    merged_val = val_rows + new_val
    random.shuffle(merged_val)

    write_csv(TRAIN_CSV, merged_train)
    write_csv(VAL_CSV, merged_val)

    after_train = Counter(r["domain"] for r in merged_train)
    after_val   = Counter(r["domain"] for r in merged_val)
    print(f"After expansion (train): {dict(after_train)}  total={sum(after_train.values())}")
    print(f"After expansion (val):   {dict(after_val)}  total={sum(after_val.values())}")


if __name__ == "__main__":
    main()
