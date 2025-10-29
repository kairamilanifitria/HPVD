import pypandoc

# Markdown content
markdown_text = r"""
# Hybrid Probabilistic Vectorized Retrieval (HPVD)

## Architecture Overview

The architecture is optimized for **enterprise-scale latency and throughput**.

- **Sparse retrieval (BM25):**  
  \( O(|q| \cdot \log |D|) \), where \( |q| \) is the query length and \( |D| \) the number of documents.

- **Dense retrieval (ANN):**  
  Sub-linear in \( |D| \) due to approximate search \( O(\log |D|) \) per query.

- **Calibration fusion:**  
  Linear in top-\( k \) documents \( O(k) \), negligible compared to retrieval.

### Latency Benchmarks
- **p50 latency:** ~50 ms per query for \( |D| \approx 10^6 \).  
- **p95 latency:** ~200 ms, depending on ANN index type and calibration complexity.

The system balances **precision, calibration accuracy, and latency**, ensuring usability in **real-time CRM scenarios** as well as **high-stakes domains like healthcare or finance**.

---

## Probabilistic Calibration and Fusion

The core innovation of HPVD lies in its **probabilistic calibration and fusion** of sparse and dense retrieval signals.

- **Joint calibration:**  
  Both BM25 and dense scores are mapped into probability distributions via calibration methods such as *isotonic regression*, *Platt scaling*, and *temperature scaling*.

- **Variational Bayesian fusion:**  
  Posterior probability of document relevance is estimated by combining calibrated scores:

  \[
  P(d \mid q) \propto P_\theta(S_{\text{sparse}} \mid q, d) \cdot P_\phi(S_{\text{dense}} \mid q, d)
  \]

- **Dynamic abstention:**  
  If posterior uncertainty exceeds a threshold, HPVD outputs **“no confident result”**, delegating retrieval to fallback mechanisms or human review.

This layer ensures that hybrid retrieval is not just additive but **confidence-aware, auditable, and compliant with EU trustworthiness requirements**.

---

## Conformal Prediction for Trustworthiness

To further enhance trustworthiness, HPVD integrates **conformal prediction** into the retrieval process.

- **Selective retrieval guarantees:**  
  Given a query \( q \), HPVD produces not just a ranked list, but a **conformal set of documents** guaranteed to contain the true relevant documents with confidence \( 1 - \alpha \).

  \[
  \mathbb{P}(d^* \in \Gamma(q)) \ge 1 - \alpha
  \]

  where \( \Gamma(q) \) is the conformal retrieval set.

- **Risk–coverage trade-offs:**  
  By adjusting the confidence threshold \( \alpha \), HPVD balances between **coverage (recall)** and **risk (error rate)**.  
  This enables *intelligent abstention*, where the system outputs **“no confident result”** when uncertainty is too high, delegating the case to human oversight or fallback pipelines.

---

## Calibration Metrics

Raw scores from BM25 and embeddings are not comparable.  
Calibration transforms them into **probabilistic confidence estimates** that reflect the true likelihood of relevance.

- **Expected Calibration Error (ECE):**  
  For retrieval, we define *retrieval ECE* over ranked documents.  
  Partition confidence scores into bins \( \{B_m\} \). Then:

  \[
  \text{ECE} = \sum_m \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
  \]

  where \( \text{acc}(B_m) \) is the empirical relevance rate and \( \text{conf}(B_m) \) the mean confidence.

- **Brier Score for top-\( k \) ranking:**  
  The Brier Score measures the mean squared error of probabilistic predictions:

  \[
  \text{BS} = \frac{1}{k} \sum_{i=1}^{k} (p_i - y_i)^2
  \]

  where \( p_i \) is the predicted probability of relevance for the \( i \)-th document in the top-\( k \), and \( y_i \in \{0, 1\} \) its ground truth.

- **Reliability diagrams for hybrid retrieval:**  
  Reliability diagrams visualize calibration across sparse–dense fusion.  
  The x-axis represents predicted confidence, and the y-axis empirical relevance probability, enabling **auditable calibration checks** for hybrid systems.

---

## Probabilistic Formulation

We formalize hybrid retrieval as a **probabilistic inference problem**.

Let \( (\mathcal{D}, \mathcal{Q}) \) denote the document and query spaces,  
with a query \( q \in \mathcal{Q} \) and candidate documents \( d \in \mathcal{D} \).

Each retrieval method provides a **score function**:

- **BM25 (sparse lexical retrieval):**  
  \( s_{\text{BM25}}(q, d) \), a deterministic score based on term frequency–inverse document frequency.

- **Dense embeddings (neural semantic retrieval):**  
  \( s_{\text{dense}}(q, d) = \langle f(q), g(d) \rangle \),  
  where \( f \) and \( g \) are neural encoders.

In HPVD, these scores are treated not as opaque values but as **random variables drawn from calibrated probability distributions:**

\[
S_{\text{BM25}} \sim P_\theta(s \mid q, d), \quad
S_{\text{dense}} \sim P_\phi(s \mid q, d)
\]

where \( \theta \) and \( \phi \) are distribution parameters learned during calibration.

The **Bayesian fusion of heterogeneous scores** is then defined as:

\[
P(d \mid q) \propto P_\theta(S_{\text{BM25}} \mid q, d) \cdot P_\phi(S_{\text{dense}} \mid q, d)
\]

This formulation ensures that BM25 and dense similarity contribute within a **unified probabilistic space**, enabling transparent interpretation and comparison.
"""

# Convert to markdown file
output_path = "/mnt/data/HPVD_Retrieval_System.md"
pypandoc.convert_text(markdown_text, "md", format="md", outputfile=output_path, extra_args=["--standalone"])

output_path
