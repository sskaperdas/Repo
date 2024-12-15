import random

import numpy as np
import tensorflow as tf
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load and preprocess the knowledge graph from the TTL file
def load_graph(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")
    return g


def preprocess_data(graph):
    entities = list(set(s for s, _, _ in graph) | set(o for _, _, o in graph))
    relations = list(set(p for _, p, _ in graph))

    entity2idx = {entity: idx for idx, entity in enumerate(entities)}
    idx2entity = {idx: entity for idx, entity in enumerate(entities)}
    relation2idx = {relation: idx for idx, relation in enumerate(relations)}
    idx2relation = {idx: relation for idx, relation in enumerate(relations)}

    num_entities = len(entities)
    num_relations = len(relations)

    triples = [(entity2idx[s], relation2idx[p], entity2idx[o]) for s, p, o in graph]

    return num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation


class MarginRankingLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.1):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        pos_scores = tf.boolean_mask(y_pred, y_true == 1)
        neg_scores = tf.boolean_mask(y_pred, y_true == 0)

        pos_scores = tf.expand_dims(pos_scores, axis=0)
        neg_scores = tf.expand_dims(neg_scores, axis=1)

        margin_matrix = self.margin - pos_scores + neg_scores
        loss = tf.reduce_sum(tf.maximum(0.0, margin_matrix))

        return loss


class NoEmbds_Model(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim, dropout_rate=0.5, l2_reg=0.01):
        super(NoEmbds_Model, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim, name='entity_embedding')
        self.relation_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim, name='relation_embedding')

        # Graph Convolution Layers
        self.conv1d_1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv1d_2 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, inputs, training=None, mask=None):
        s_embed = self.entity_embeddings(inputs[:, 0])
        p_embed = self.relation_embeddings(inputs[:, 1])
        o_embed = self.entity_embeddings(inputs[:, 2])

        concatenated_embed = tf.concat([s_embed, p_embed, o_embed], axis=1)
        x = tf.expand_dims(concatenated_embed, axis=-1)  # Add channel dimension for Conv1D

        # Convolutional layers
        x = self.conv1d_1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.conv1d_2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        # Output layer
        prediction = self.dense2(x)

        return prediction


# Define diverse negative sampling function
def diverse_negative_sampling(triples, num_entities, num_relations, corruption_strength='medium', batch_size=1000):
    negative_triples = []
    triple_set = set(triples)

    for i in range(0, len(triples), batch_size):
        batch_triples = triples[i:i + batch_size]
        batch_negatives = []

        for triple in batch_triples:
            s, p, o = triple

            if corruption_strength == 'low':
                # Only replace subject
                s_new = np.random.randint(num_entities)
                while (s_new, p, o) in triple_set:
                    s_new = np.random.randint(num_entities)
                batch_negatives.append((s_new, p, o))

            elif corruption_strength == 'medium':
                # Replace both subject and object
                s_new = np.random.randint(num_entities)
                o_new = np.random.randint(num_entities)
                while (s_new, p, o) in triple_set or (s, p, o_new) in triple_set:
                    s_new = np.random.randint(num_entities)
                    o_new = np.random.randint(num_entities)
                batch_negatives.append((s_new, p, o))
                batch_negatives.append((s, p, o_new))

            elif corruption_strength == 'high':
                # Replace subject, object, and relation
                s_new = np.random.randint(num_entities)
                p_new = np.random.randint(num_relations)
                o_new = np.random.randint(num_entities)
                while (s_new, p, o) in triple_set or (s, p_new, o) in triple_set or (s, p, o_new) in triple_set:
                    s_new = np.random.randint(num_entities)
                    p_new = np.random.randint(num_relations)
                    o_new = np.random.randint(num_entities)
                batch_negatives.append((s_new, p, o))
                batch_negatives.append((s, p_new, o))
                batch_negatives.append((s, p, o_new))

        negative_triples.extend(batch_negatives)

    return negative_triples


# Load and preprocess the knowledge graph
graph_file_path = "statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Generate negative examples

negative_triples = diverse_negative_sampling(triples, num_entities, num_relations, corruption_strength='high')
all_triples = np.vstack((triples, np.array(negative_triples)))
labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

from sklearn.model_selection import StratifiedShuffleSplit

stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_index, temp_index = next(stratified_splitter.split(all_triples, labels))

X_train, X_temp, y_train, y_temp = all_triples[train_index], all_triples[temp_index], labels[train_index], labels[
    temp_index]

stratified_splitter_test_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

test_index, val_index = next(stratified_splitter_test_val.split(X_temp, y_temp))

X_test, X_val, y_test, y_val = X_temp[test_index], X_temp[val_index], y_temp[test_index], y_temp[val_index]

# Create and compile the model
embedding_dim = 400
initial_learning_rate = 0.00005
num_epochs = 15
batch_size = 128

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

# Create and compile the model with
model = NoEmbds_Model(num_entities, num_relations, embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    np.array(X_train), y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(np.array(X_val), y_val),
    callbacks=[early_stopping, lr_schedule],
    verbose=1
)

# Evaluate the model on the test set
test_scores = model.predict(np.array(X_test))
y_pred = (test_scores > 0.5).astype(int)
y_true = y_test.astype(int)

# Calculate evaluation metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, test_scores)
pr_auc = average_precision_score(y_true, test_scores)

# Display evaluation metrics
print("\nEvaluation Metrics:")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'PR AUC: {pr_auc:.4f}')

# Plot training vs validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
