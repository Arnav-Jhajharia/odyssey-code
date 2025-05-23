import random, re, itertools, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Detect CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load cluster data
cluster_file = 'UnderlyingClusters.txt'
print(f'Using file: {cluster_file}')

def parse_cluster_file(path):
    clusters = {}
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^CLUSTER\s+(\d+)', line, re.I)
        if m:
            current = int(m.group(1))
            clusters[current] = []
        else:
            if current is None:
                raise ValueError('File format error: missing CLUSTER header')
            clusters[current].append(line.upper())
    return clusters

clusters = parse_cluster_file(cluster_file)
print(f'Parsed {len(clusters)} clusters, total sequences: {sum(map(len, clusters.values()))}')

# Build contrastive training pairs
def build_examples(clusters):
    examples = []
    for seqs in clusters.values():
        for s in seqs:
            positive = random.choice([x for x in seqs if x != s] or [s])
            examples.append(InputExample(texts=[s, positive]))
    return examples

examples = build_examples(clusters)
train_examples, _ = train_test_split(examples, test_size=0.1, random_state=SEED)
print('Training pairs:', len(train_examples))

# Backbone model (replace with DNABERT manually if needed)
backbone = models.Transformer('bert-base-uncased', max_seq_length=512)
pooling = models.Pooling(
    backbone.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[backbone, pooling])

# Move to GPU if available
model.to(device)

# DataLoader with pin_memory for CUDA
loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=32,
    pin_memory=(device == 'cuda')
)

# Loss function
loss_fn = losses.MultipleNegativesRankingLoss(model)

# Train model using AMP if on CUDA
model.fit(
    train_objectives=[(loader, loss_fn)],
    epochs=10,             # Increase for better quality
    warmup_steps=100,
    use_amp=(device == 'cuda'),
    show_progress_bar=True,
    output_path='dna-embeddings-model'
)

# Encode all sequences
all_seqs = list(itertools.chain.from_iterable(clusters.values()))
embeddings = model.encode(
    all_seqs,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=device  # Ensure encoding also uses CUDA
)

torch.save(embeddings, 'dna_embeddings.pt')
print('Saved dna_embeddings.pt with shape', embeddings.shape)

