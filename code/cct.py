import pandas as pd

def load_plant_knowledge_data(filepath):
    df = pd.read_csv(filepath)  # Assumes comma-separated values
    if 'Informant' in df.columns:
        df = df.drop(columns=['Informant'])
    return df.values

# Load data
data = load_plant_knowledge_data("cct-midterm/data/plant_knowledge.csv")
N, M = data.shape  # Number of informants and items

import pymc as pm
import numpy as np
import arviz as az

with pm.Model() as cct_model:
    # Prior for informant competence (D): 0.5 to 1 range
    D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)
  

    # Prior for consensus answers (Z): Binary (0 or 1)
    Z = pm.Bernoulli("Z", p=0.5, shape=M)  

    # Reshape 
    D_reshaped = D[:, None]  # (N, 1)
    Z_reshaped = Z[None, :]  # (1, M)

    # Define pij: the probability each informant gets each question right
    p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)  # shape: (N, M)

    # Likelihood
    X = pm.Bernoulli("X", p=p, observed=data)
with cct_model:
    trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9, random_seed=42)

summary = az.summary(trace, var_names=["D", "Z"])
print(summary)
az.plot_trace(trace, var_names=["D", "Z"])
az.plot_pair(trace, var_names=["D", "Z"], kind='scatter', divergences=True)

# Posterior means of informant competence, I used chatGPT to help me refine and correct this part of the code
competence_means = trace.posterior["D"].mean(dim=["chain", "draw"]).values
print("Competence Estimates:")
for i, val in enumerate(competence_means):
    print(f"Informant {i+1}: {val:.3f}")

az.plot_posterior(trace, var_names=["D"])

# Posterior mean probabilities for each consensus answer
consensus_means = trace.posterior["Z"].mean(dim=["chain", "draw"]).values

print("\nConsensus Answer Estimates:")
for i, val in enumerate(consensus_means):
    print(f"PQ{i+1}: Posterior Mean = {val:.3f}")

#Consensus answer key based on CCT model
consensus_probs = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
consensus_answers = (consensus_probs > 0.5).astype(int)

print("Consensus Answer Key:")
print(consensus_answers)

az.plot_posterior(trace, var_names=["Z"])

# Naive aggregation for majority
majority_vote = (data.mean(axis=0) > 0.5).astype(int)

print("Naive Majority Vote:")
print(majority_vote)

# Compare with CCT model results
diff = consensus_answers != majority_vote
print(f"Items with disagreement: {np.where(diff)[0]}")
