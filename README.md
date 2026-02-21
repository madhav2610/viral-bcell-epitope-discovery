🦠 Viral B-Cell Epitope Discovery: A Deep Learning Motif Scanner


🎯 The Objective

B-cell epitopes are the specific regions of a virus that are recognized by the human immune system. Traditional epitope discovery relies on expensive and time-consuming wet-lab assays. The goal of this project was to build a computationally lightweight deep learning pipeline that learns from historical viral datasets (SARS and curated B-cell epitope data) to predict potential immunogenic regions on previously unseen SARS-CoV-2 sequences.

🏗️ The Architecture (1D CNN–RNN Hybrid)

Predicting immunogenic regions from 1D amino acid sequences—without explicit 3D structural information—is notoriously difficult. To address this, a hybrid neural architecture was designed:

Motif Scanner (Conv1D + MaxPool1D): Acts as a local pattern detector, scanning sequences for biologically meaningful k-mer motifs representing physicochemical and secondary-structure–associated patterns.

Sequence Context Reader (Bidirectional GRU): Processes extracted motifs bidirectionally to capture broader contextual dependencies along the protein sequence.

Overfitting Safeguards: Heavy Dropout (0.4) and EarlyStopping monitored on validation ROC-AUC were used to stabilize learning under class imbalance.

⚖️ Handling Biological Imbalance

In viral datasets, true epitopes are heavily outnumbered by non-epitopes, causing naïve models to favor negative predictions. To address this, arbitrary undersampling was avoided and inverse-frequency class weights were computed dynamically ({0: 0.686, 1: 1.844}). This biases the learning process toward minimizing false negatives, which is critical in biological screening tasks.

🔬 Results: SARS-CoV-2 Blind Screening

Note: These predictions represent computational prioritization and not experimental validation.

The trained model was applied blindly to SARS-CoV-2 peptide sequences. Despite lacking explicit structural information, the model highlighted multiple regions previously reported in the literature as immunogenic, including:

Receptor Binding Domain (RBD): Motif clusters such as GYQPYR / YQPYRV associated with viral entry mechanisms.

Spike Protein Fragments: Core regions overlapping known neutralizing domains.

Nucleocapsid Domains: Proline-rich regions linked to N-protein immunogenicity.
