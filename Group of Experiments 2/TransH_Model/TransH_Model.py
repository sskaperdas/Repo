import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    matthews_corrcoef, confusion_matrix
from rdflib import Graph, URIRef
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard


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


def compute_inverse_relation_frequency(triples, num_relations):
    # Count the frequency of each relation
    relation_frequency = np.zeros(num_relations)
    for _, p, _ in triples:
        relation_frequency[p] += 1

    # Compute the inverse of the frequencies
    inverse_relation_frequency = 1 / (relation_frequency + 1)  # Add 1 to avoid division by zero

    return inverse_relation_frequency


# Define the triple classification TransE model with the dense architecture
class TransH_Model(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransH_Model, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(
            num_entities, embedding_dim, name='entity_embedding')
        self.relation_embeddings = tf.keras.layers.Embedding(
            num_relations, embedding_dim, name='relation_embedding')

        # Add embedding dimensions for entity degree and inverse relation frequency
        self.in_degree = tf.keras.layers.Embedding(
            num_entities, 1, name='in_degree_embedding')
        self.out_degree = tf.keras.layers.Embedding(
            num_entities, 1, name='out_degree_embedding')
        self.inverse_relation_frequency_embedding = tf.keras.layers.Embedding(
            num_relations, 1, name='inverse_relation_frequency_embedding',
            embeddings_initializer=tf.keras.initializers.Constant(inverse_relation_frequency),
            trainable=False
        )

        # Hyperplane embeddings for TransH
        self.hyperplane_embeddings = tf.keras.layers.Embedding(
            num_relations, embedding_dim * embedding_dim, name='hyperplane_embedding')

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

        inverse_relation_frequency_embed = self.inverse_relation_frequency_embedding(inputs[:, 1])
        s_embed += inverse_relation_frequency_embed
        o_embed += inverse_relation_frequency_embed

        # Project entities onto hyperplanes for TransH
        hyperplane_embed = self.hyperplane_embeddings(inputs[:, 1])
        hyperplane_embed = tf.reshape(hyperplane_embed, (-1, s_embed.shape[1], s_embed.shape[1]))

        s_projected = tf.linalg.matvec(hyperplane_embed, s_embed)
        o_projected = tf.linalg.matvec(hyperplane_embed, o_embed)

        # Normalize the projected entity embeddings
        s_projected = self.normalization(s_projected)
        o_projected = self.normalization(o_projected)

        predicted_o_embed = s_projected + p_embed
        concat_embed = tf.concat([s_embed, p_embed, o_projected, predicted_o_embed], axis=1)

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
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
graph_file_path = "statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Compute entity degrees
entity_degrees = compute_entity_degrees(triples)

# Generate negative examples
negative_triples = [(s, p, (o + np.random.randint(1, num_entities)) % num_entities) for s, p, o in triples]
negative_triples_with_degrees = np.array([
    (s, p, o, entity_degrees[s], entity_degrees[o]) for s, p, o in negative_triples
])
positive_triples_with_degrees = np.array([
    (s, p, o, entity_degrees[s], entity_degrees[o]) for s, p, o in triples
])
all_triples_with_degrees = np.vstack((positive_triples_with_degrees, negative_triples_with_degrees))

labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

# Shuffle and split data into train, test, and validation sets
X_train_val, X_test, y_train_val, y_test = train_test_split(all_triples_with_degrees, labels, test_size=0.2, stratify=labels,
                                                            random_state=42)

inverse_relation_frequency = compute_inverse_relation_frequency(triples, num_relations)

stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Create and compile the model
embedding_dim = 512
num_epochs = 25
batch_size = 256

for fold, (train_index, val_index) in enumerate(stratified_kfold.split(X_train_val, y_train_val)):
    print(f"\nTraining on Fold {fold + 1}")

    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

    # Create and compile the model for each fold
    model = TransH_Model(num_entities, num_relations, embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                  metrics=['accuracy'])

    plateau_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.6,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Create a directory for storing TensorBoard logs
    log_dir = "logs/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model for the current fold
    history = model.fit(
        np.array(X_train), y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(np.array(X_val), y_val),
        callbacks=[early_stopping, plateau_scheduler, tensorboard_callback],
        verbose=1
    )

# After training on all folds, evaluate on the test set
test_scores = model.predict(np.array(X_test))
y_test_pred = (test_scores > 0.5).astype(int)
y_test_true = y_test.astype(int)

# Calculate and display evaluation metrics for the test set
precision_test = precision_score(y_test_true, y_test_pred)
recall_test = recall_score(y_test_true, y_test_pred)
f1_test = f1_score(y_test_true, y_test_pred)
roc_auc_test = roc_auc_score(y_test_true, test_scores)
pr_auc_test = average_precision_score(y_test_true, test_scores)
mcc_test = matthews_corrcoef(y_test_true, y_test_pred)

print("\nTest Metrics:")
print(f'Precision: {precision_test:.4f}')
print(f'Recall: {recall_test:.4f}')
print(f'F1 Score: {f1_test:.4f}')
print(f'ROC AUC: {roc_auc_test:.4f}')
print(f'PR AUC: {pr_auc_test:.4f}')
print(f'MCC: {mcc_test:.4f}')
