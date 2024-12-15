import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

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


def compute_entity_degrees(triples):
    entity_degrees = {}

    for s, _, _ in triples:
        if s in entity_degrees:
            entity_degrees[s] += 1
        else:
            entity_degrees[s] = 1

    for _, _, o in triples:
        if o in entity_degrees:
            entity_degrees[o] += 1
        else:
            entity_degrees[o] = 1

    return entity_degrees


# Define the triple classification DistMult model with the dense architecture
class DistMult_Model(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult_Model, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(
            num_entities, embedding_dim, name='entity_embedding')
        self.relation_embeddings = tf.keras.layers.Embedding(
            num_relations, embedding_dim, name='relation_embedding')

        # Add embedding dimensions for entity degree and inverse relation frequency
        self.in_degree = tf.keras.layers.Embedding(
            num_entities, 1, name='in_degree_embedding')
        self.out_degree = tf.keras.layers.Embedding(
            num_entities, 1, name='out_degree_embedding')
        self.inverse_relation_frequency = tf.keras.layers.Embedding(
            num_relations, 1, name='inverse_relation_frequency')

        # Normalize the entity embeddings
        self.normalization = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(
            5012, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(
            2048, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.dense3 = tf.keras.layers.Dense(
            1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout3 = tf.keras.layers.Dropout(0.5)

        self.dense4 = tf.keras.layers.Dense(
            512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout4 = tf.keras.layers.Dropout(0.5)

        self.dense5 = tf.keras.layers.Dense(
            256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout5 = tf.keras.layers.Dropout(0.5)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        s_embed = self.entity_embeddings(inputs[:, 0])
        p_embed = self.relation_embeddings(inputs[:, 1])
        o_embed = self.entity_embeddings(inputs[:, 2])

        # Add in-degree, out-degree, and inverse relation frequency embedding
        in_degree_embed = self.in_degree(inputs[:, 0])
        s_embed += in_degree_embed

        out_degree_embed = self.out_degree(inputs[:, 2])
        o_embed += out_degree_embed

        inverse_relation_frequency_embed = self.inverse_relation_frequency(inputs[:, 1])
        s_embed += inverse_relation_frequency_embed
        o_embed += inverse_relation_frequency_embed

        # Normalize the entity embeddings
        s_embed = self.normalization(s_embed)
        o_embed = self.normalization(o_embed)

        # DistMult scoring function
        score = tf.reduce_sum(s_embed * p_embed * o_embed, axis=1, keepdims=True)

        concat_embed = tf.concat([s_embed, p_embed, o_embed, score], axis=1)

        x = self.dense1(concat_embed)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        x = self.dense4(x)
        x = self.dropout4(x, training=training)

        x = self.dense5(x)
        x = self.dropout5(x, training=training)

        flattened = self.flatten(x)
        prediction = self.dense(flattened)

        return prediction


# Load and preprocess the knowledge graph
graph_file_path = "statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Compute entity degrees
entity_degrees = compute_entity_degrees(triples)

# Bernoulli Negative Sampling
negative_triples = []
for s, p, o in triples:
    coin_toss = np.random.randint(2)
    if coin_toss == 0:
        s_neg = np.random.choice(num_entities)
        negative_triples.append((s_neg, p, o))
    else:
        o_neg = np.random.choice(num_entities)
        negative_triples.append((s, p, o_neg))

all_triples = np.vstack((np.array(triples), np.array(negative_triples)))
labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

# Shuffle and split data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(all_triples, labels, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and compile the model
embedding_dim = 256
num_epochs = 30
batch_size = 256

model = DistMult_Model(num_entities, num_relations, embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
              metrics=['accuracy'])


plateau_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.6,  # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-7,  # Lower bound on the learning rate
    verbose=1  # Optional: Display learning rate updates
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Create a directory for storing TensorBoard logs
log_dir = "logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(
    np.array(X_train), y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(np.array(X_val), y_val),
    callbacks=[early_stopping, plateau_scheduler, tensorboard_callback],  # Include lr_callback in callbacks
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
mcc = matthews_corrcoef(y_true, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Display evaluation metrics
print("\nEvaluation Metrics:")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'PR AUC: {pr_auc:.4f}')
print(f'MCC: {mcc:.4f}')

# Display Confusion Matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot training vs validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
