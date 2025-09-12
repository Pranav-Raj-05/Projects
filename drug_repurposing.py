# ============================================================
# Drug Repurposing with Knowledge Graphs (Advanced Analysis)
# ============================================================

import logging
from pykeen.training.callbacks import TrainingCallback
from pykeen.training import SLCWATrainingLoop

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import plotly.express as px
import plotly.io as pio
import networkx as nx
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
logging.getLogger('pykeen').setLevel(logging.ERROR)

# --- Configuration ---
TRIPLES_FILE = "triples_4k.tsv" 
OUTPUT_DIR = "pykeen_outputs_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model & Training Hyperparameters
MODEL_NAME = "ComplEx" 
EMBEDDING_DIM = 100
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Analysis & Visualization Parameters
TOP_K_PREDICTIONS = 3 
EXAMPLE_ENTITIES_TO_PLOT = ['ACE', 'Ibuprofen', 'Simvastatin'] 

# --- 1. Data Loading ---
# Ensure the triples file exists, create a dummy if not
if not os.path.exists(TRIPLES_FILE):
    print(f"Error: Triples file not found at '{TRIPLES_FILE}'")
    dummy_data = {
        'head': ['drugA', 'drugB', 'drugA'], 'relation': ['treats', 'treats', 'causes_side_effect'],
        'tail': ['diseaseX', 'diseaseY', 'side_effectZ']
    }
    pd.DataFrame(dummy_data).to_csv(TRIPLES_FILE, sep='\t', index=False, header=False)
    print(f"Created a dummy '{TRIPLES_FILE}' to proceed.")

# Load triples using PyKEEN's TriplesFactory
tf = TriplesFactory.from_path(TRIPLES_FILE)
print(f"✅ Loaded {tf.num_triples} triples with {tf.num_entities} entities and {tf.num_relations} relations.\n")

# --- 2. Live Training Monitoring Callback ---
class EpochMetricsLogger(TrainingCallback):
    """Logs loss + MRR + Hits@K after each epoch for real-time monitoring."""
    def __init__(self, triples_factory, **kwargs):
        super().__init__(**kwargs)
        self.epoch_count = 0
        self.tf = triples_factory
        self.evaluator = RankBasedEvaluator()

    def on_epoch_end(self, training_loop: SLCWATrainingLoop):
        self.epoch_count += 1
        loss = training_loop.losses[-1].item()
        training_loop.model.eval()
        try:
            # Evaluate on the full training set to monitor progress
            eval_result = self.evaluator.evaluate(
                model=training_loop.model, mapped_triples=self.tf.mapped_triples,
                additional_filter_triples=[self.tf.mapped_triples], batch_size=BATCH_SIZE,
            )
            mrr = eval_result.get_metric("mean_reciprocal_rank")
            hits1 = eval_result.get_metric("hits_at_1")
            hits10 = eval_result.get_metric("hits_at_10")
            print(f"Epoch {self.epoch_count}/{NUM_EPOCHS} - loss={loss:.6f}, "
                  f"MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@10={hits10:.4f}")
        finally:
            training_loop.model.train()

# --- 3. Model Training ---
print(f"Training {MODEL_NAME} model with live MRR/Hits monitoring...")
epoch_logger = EpochMetricsLogger(tf)

result = pipeline(
    training=tf, testing=tf, model=MODEL_NAME,
    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
    training_kwargs=dict(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[epoch_logger]),
    optimizer="Adam", optimizer_kwargs=dict(lr=LEARNING_RATE), random_seed=42,
)
print("✅ Model training complete.\n")

# --- 4. Per-Relation Performance Analysis ---
print("📊 Analyzing performance for each relation type...")
per_relation_results = []
evaluator = RankBasedEvaluator()
all_triples_df = pd.DataFrame(tf.triples, columns=['head', 'relation', 'tail'])

for relation in tf.relation_id_to_label.values():
    try:
        relation_df = all_triples_df[all_triples_df['relation'] == relation]
        if relation_df.empty:
            print(f"Skipping relation '{relation}' as it has no triples.")
            continue
        
        relation_tf = TriplesFactory.from_labeled_triples(
            relation_df.values,
            entity_to_id=tf.entity_to_id,
            relation_to_id=tf.relation_to_id,
        )

        metrics = evaluator.evaluate(
            model=result.model, mapped_triples=relation_tf.mapped_triples,
            additional_filter_triples=[tf.mapped_triples], batch_size=BATCH_SIZE
        ).to_dict()['both']['realistic']
        metrics['relation'] = relation
        per_relation_results.append(metrics)
    except Exception as e:
        print(f"Could not evaluate relation '{relation}': {e}")

if per_relation_results:
    df_per_relation = pd.DataFrame(per_relation_results)
    
    desired_metrics = ['mean_reciprocal_rank', 'hits_at_1', 'hits_at_3', 'hits_at_10']
    available_metrics = [metric for metric in desired_metrics if metric in df_per_relation.columns]
    
    if not available_metrics:
        print("⚠️ No standard rank-based metrics found in per-relation results to plot.")
    else:
        plot_columns = ['relation'] + available_metrics
        df_plot = df_per_relation[plot_columns].melt(
            id_vars='relation', var_name='Metric', value_name='Score'
        )
        fig = px.bar(
            df_plot, x='relation', y='Score', color='Metric', barmode='group',
            title='Per-Relation Performance Metrics',
            labels={'Score': 'Metric Score', 'relation': 'Relation Type'}
        )
        fig.update_layout(xaxis_title="", yaxis_title="Score", legend_title="Metric")
        fig.write_html(os.path.join(OUTPUT_DIR, "per_relation_performance.html"))
        print("✅ Per-relation performance chart saved.")

# --- 5. Prediction Generation ---
def get_top_predictions(model, head, relation, factory, k):
    """Scores a (head, relation, ?) query and returns top k tails."""
    try:
        head_id = factory.entity_to_id[head]
        relation_id = factory.relation_to_id[relation]
    except KeyError: return []
    all_tail_ids = torch.arange(factory.num_entities, device=model.device)
    batch = torch.stack([
        torch.full_like(all_tail_ids, head_id),
        torch.full_like(all_tail_ids, relation_id),
        all_tail_ids
    ], dim=1)
    with torch.no_grad():
        scores = model.score_hrt(batch).cpu().numpy().flatten()
    top_indices = np.argsort(scores)[::-1][:k]
    return [(factory.entity_id_to_label[i], scores[i]) for i in top_indices]

print(f"\nGenerating top {TOP_K_PREDICTIONS} predictions for repurposing (relation: 'treats')...")
all_results = []
for entity in tqdm(tf.entity_id_to_label.values(), desc="Predicting for all entities"):
    preds = get_top_predictions(result.model, entity, "treats", tf, k=TOP_K_PREDICTIONS)
    for rank, (tail, score) in enumerate(preds, 1):
        all_results.append([entity, "treats", tail, score, rank])

df_predictions = pd.DataFrame(all_results, columns=["Head", "Relation", "Predicted_Tail", "Score", "Rank"])
df_predictions.to_csv(os.path.join(OUTPUT_DIR, "all_entity_predictions.csv"), index=False)
print(f"✅ All predictions saved. Top {min(3, len(df_predictions))} results:")
print(df_predictions.head(3))

# --- 6. Visualization of Predictions & Embeddings ---
if 'treats' in tf.relation_to_id:
    for entity in EXAMPLE_ENTITIES_TO_PLOT:
        if entity in tf.entity_to_id:
            preds = get_top_predictions(result.model, entity, "treats", tf, k=TOP_K_PREDICTIONS)
            df_preds = pd.DataFrame(preds, columns=['Predicted Disease/Symptom', 'Plausibility Score'])
            fig = px.bar(df_preds, x='Plausibility Score', y='Predicted Disease/Symptom', orientation='h',
                         title=f"Top {TOP_K_PREDICTIONS} 'treats' Predictions for: {entity}")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            fig.write_html(os.path.join(OUTPUT_DIR, f"predictions_{entity}.html"))
    print(f"\n✅ Top-{TOP_K_PREDICTIONS} prediction charts for example entities saved.")

print("\n🎨 Visualizing entity embeddings in 2D space...")
# For ComplEx, embeddings are complex. We concatenate real and imaginary parts.
entity_embeddings_complex = result.model.entity_representations[0](indices=None).detach().cpu().numpy()
entity_embeddings_real = np.concatenate(
    [entity_embeddings_complex.real, entity_embeddings_complex.imag],
    axis=1,
)

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, tf.num_entities - 1))
embeddings_2d = tsne.fit_transform(entity_embeddings_real)

df_embeddings = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
df_embeddings['label'] = list(tf.entity_id_to_label.values())

def get_entity_type(label):
    if any(s in label.lower() for s in ['drug', 'statin', 'mab', 'nib', 'pril', 'sartan', 'ine']):
        return 'Drug/Compound'
    if any(s in label.lower() for s in ['cancer', 'disease', 'itis', 'syndrome', 'infection']):
        return 'Disease/Symptom'
    return 'Gene/Protein'
df_embeddings['type'] = df_embeddings['label'].apply(get_entity_type)

fig = px.scatter(
    df_embeddings, x='x', y='y', text='label', color='type',
    title='2D t-SNE Visualization of Entity Embeddings',
    labels={'color': 'Entity Type'},
    hover_data={'x': False, 'y': False, 'label': True}
)
fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_layout(showlegend=True, xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2")
fig.write_html(os.path.join(OUTPUT_DIR, "embedding_visualization_tsne.html"))
print("✅ t-SNE embedding visualization saved.")

# --- 7. Interactive Knowledge Graph Network Visualization ---
print("\n🔗 Building interactive knowledge graph visualization...")
G = nx.DiGraph()
for h, r, t in tf.triples:
    G.add_node(h, type="Entity")
    G.add_node(t, type="Entity")
    G.add_edge(h, t, relation=r)

pos = nx.spring_layout(G, seed=42, k=0.8)
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
edge_trace = dict(type="scattergl", x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="none", mode="lines")

node_x, node_y, node_text, node_color = [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x); node_y.append(y)
    node_text.append(f"{node}<br># of connections: {G.degree[node]}"); node_color.append(G.degree[node])
node_trace = dict(
    type="scattergl", x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
    marker=dict(showscale=True, colorscale="Viridis", reversescale=True, color=node_color, size=12,
                colorbar=dict(thickness=15, title="Node Degree"), line_width=2)
)

fig_dict = dict(data=[edge_trace, node_trace], layout=dict(
    title="Interactive Knowledge Graph Network", showlegend=False, hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
))
pio.write_html(fig_dict, os.path.join(OUTPUT_DIR, "knowledge_graph_network.html"))
print("✅ Interactive Knowledge Graph saved.")

# --- 8. Advanced Evaluation Visualizations ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

print("\n📊 Generating additional evaluation visualizations...")

# Get top-1 prediction for each triple in the original data
gold_triples = tf.triples.tolist()
predicted_triples = []
for h, r, t in gold_triples:
    preds = get_top_predictions(result.model, h, r, tf, k=1)
    if preds:
        predicted_triples.append((h, r, preds[0][0]))
    else:
        # Handle cases where prediction fails for a given head/relation
        predicted_triples.append((h, r, "<None>"))

y_true = [t for _, _, t in gold_triples]
y_pred = [t for _, _, t in predicted_triples]
labels = sorted(set(y_true) | set(y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
plt.title("Confusion Matrix of Predicted vs Actual Tails")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print("✅ Confusion matrix saved.")

# Precision, Recall, F1-Score
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, zero_division=0
)
df_pr = pd.DataFrame({
    "Label": labels, "Precision": precision, "Recall": recall, "F1": f1, "Support": support
})
df_pr.to_csv(os.path.join(OUTPUT_DIR, "precision_recall_f1.csv"), index=False)

# Precision vs Recall Bar Plot
fig, ax = plt.subplots(figsize=(12, 6))
df_pr.plot(x="Label", y=["Precision", "Recall"], kind="bar", ax=ax)
plt.title("Precision vs Recall per Label")
plt.ylabel("Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_vs_recall.png"), dpi=300)
plt.close()
print("✅ Precision vs Recall plot saved.")

# P/R/F1 Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_pr.set_index("Label")[["Precision", "Recall", "F1"]],
            annot=True, cmap="viridis", fmt=".2f", linewidths=0.5, ax=ax)
plt.title("Summary Heatmap of Precision, Recall, F1 per Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "summary_heatmap.png"), dpi=300)
plt.close()
print("✅ Summary heatmap saved.")

# --- 9. Summary & Report-Ready Visualizations ---
if not df_predictions.empty:
    df_top3 = df_predictions.head(3)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    tbl = ax.table(cellText=df_top3.values, colLabels=df_top3.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.title("Top 3 Predictions")
    plt.savefig(os.path.join(OUTPUT_DIR, "top3_predictions.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Top-3 predictions table saved as PNG.")

if not df_predictions.empty:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_predictions["Score"], kde=True, bins=30)
    plt.title("Distribution of Prediction Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution.png"), dpi=300)
    plt.close()
    print("✅ Score distribution plot saved.")

if not df_pr.empty:
    df_pr_sorted = df_pr.sort_values("F1", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Label", y="F1", data=df_pr_sorted)
    plt.xticks(rotation=90)
    plt.title("Relations Ranked by F1-score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "relations_ranked_by_f1.png"), dpi=300)
    plt.close()
    print("✅ Relations ranked by F1-score plot saved.")

# Interactive P/R/F1 plots
if not df_pr.empty:
    fig = px.bar(
        df_pr.melt(id_vars="Label", value_vars=["Precision", "Recall"]),
        x="Label", y="value", color="variable", barmode="group",
        title="Interactive Precision vs Recall per Label",
        labels={"value": "Score", "variable": "Metric"}
    )
    fig.update_layout(xaxis_title="Label", yaxis_title="Score", legend_title="Metric")
    fig.write_html(os.path.join(OUTPUT_DIR, "precision_vs_recall_interactive.html"))
    print("✅ Interactive Precision vs Recall chart saved (HTML).")

if not df_pr.empty:
    df_melted = df_pr.melt(id_vars="Label", value_vars=["Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
    fig = px.imshow(
        df_pr.set_index("Label")[["Precision", "Recall", "F1"]].T,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Interactive Summary Heatmap (Precision / Recall / F1)"
    )
    fig.update_layout(
        xaxis_title="Label",
        yaxis_title="Metric",
        coloraxis_colorbar=dict(title="Score"),
    )
    fig.write_html(os.path.join(OUTPUT_DIR, "summary_heatmap_interactive.html"))
    print("✅ Interactive summary heatmap saved (HTML).")


print(f"\n🎉 Script finished successfully. All outputs are in the '{OUTPUT_DIR}' directory.")
