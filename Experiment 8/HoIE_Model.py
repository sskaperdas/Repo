import random

import numpy as np
import tensorflow as tf
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    matthews_corrcoef
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)
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


def combine_hole_embeddings(embed_real, embed_img):
    # Circular correlation for HolE
    combined_real = embed_real * embed_img
    combined_img = embed_real * embed_img
    return tf.concat([combined_real, combined_img], axis=-1)


class GATLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, output_dim, activation='relu', dropout_rate=0.3):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernels = [self.add_weight('kernel_head_{}'.format(i),
                                        shape=(input_dim, self.output_dim),
                                        initializer='glorot_uniform',
                                        trainable=True)
                        for i in range(self.num_heads)]
        self.biases = [self.add_weight('bias_head_{}'.format(i),
                                       shape=(self.output_dim,),
                                       initializer='zeros',
                                       trainable=True)
                       for i in range(self.num_heads)]

    def call(self, inputs, training=None):
        heads = [tf.matmul(inputs, kernel) + bias for kernel, bias in zip(self.kernels, self.biases)]
        outputs = tf.concat(heads, axis=-1)
        outputs = self.activation(outputs)
        outputs = tf.keras.layers.Dropout(self.dropout_rate)(outputs, training=training)
        return outputs


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_regularizer=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]), self.units],
                                      initializer=tf.keras.initializers.GlorotUniform(),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight("bias",
                                        shape=[self.units],
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        else:
            self.bias = None

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.bias is not None:
            output += self.bias
        if self.activation is not None:
            output = tf.keras.activations.get(self.activation)(output)

        # Add residual connection
        output += inputs
        return output


class GAT_GCN_ComplexModelWithHoIE(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim, hoie_embedding_dim, dropout_rate=0.5, l2_reg=0.01):
        super(GAT_GCN_ComplexModelWithHoIE, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim, name='entity_embedding')
        self.hoie_embeddings = tf.keras.layers.Embedding(num_entities, hoie_embedding_dim * 2,
                                                         name='hoie_embedding')  # *2 for real and imaginary parts
        self.relation_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim, name='relation_embedding')

        self.hoie_embedding_dim = hoie_embedding_dim  # Added initialization of hoie_embedding_dim

        # Additional layers for computing hoIE embeddings
        self.hoie_dense1 = tf.keras.layers.Dense(hoie_embedding_dim, activation='relu')
        self.hoie_dense2 = tf.keras.layers.Dense(hoie_embedding_dim, activation='relu')

        self.conv1d_1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv1d_2 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.gcn_layer1 = GCNLayer(400, activation='relu')  # Adjusted output dimension to 400

        # New GAT layer
        self.gat_layer = GATLayer(num_heads=4, output_dim=64, activation='relu', dropout_rate=dropout_rate)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, inputs, training=None, mask=None):
        s_idx, p_idx, o_idx = inputs[:, 0], inputs[:, 1], inputs[:, 2]

        s_embed = self.entity_embeddings(s_idx)
        p_embed = self.relation_embeddings(p_idx)
        o_embed = self.entity_embeddings(o_idx)

        # Compute hoIE embeddings
        hoie_embed = self.hoie_embeddings(s_idx)
        hoie_real, hoie_img = hoie_embed[:, :self.hoie_embedding_dim], hoie_embed[:, self.hoie_embedding_dim:]
        s_hoie = self.hoie_dense2(self.hoie_dense1(combine_hole_embeddings(hoie_real, hoie_img)))

        hoie_embed = self.hoie_embeddings(o_idx)
        hoie_real, hoie_img = hoie_embed[:, :self.hoie_embedding_dim], hoie_embed[:, self.hoie_embedding_dim:]
        o_hoie = self.hoie_dense2(self.hoie_dense1(combine_hole_embeddings(hoie_real, hoie_img)))

        # Graph Convolution Layer
        s_gcn = self.gcn_layer1(s_embed)
        p_gcn = self.gcn_layer1(p_embed)
        o_gcn = self.gcn_layer1(o_embed)

        # GAT Layer
        s_gat = self.gat_layer(s_embed)
        p_gat = self.gat_layer(p_embed)
        o_gat = self.gat_layer(o_embed)

        # Modify the concatenation to include hoIE embeddings and GCN/GAT embeddings
        concatenated_embed = tf.concat(
            [s_embed, p_embed, o_embed, s_hoie, o_hoie, s_gcn, p_gcn, o_gcn, s_gat, p_gat, o_gat], axis=1)

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

from collections import Counter
from scipy.stats import rankdata

# Define diverse negative sampling function
def popularity_based_sampling(triples, num_entities, num_relations, batch_size=1000):
    entity_counter = Counter(triple[0] for triple in triples)
    relation_counter = Counter(triple[1] for triple in triples)

    # Rank entities and relations based on their frequency
    entity_ranks = rankdata([-entity_counter[entity] for entity in range(num_entities)])
    relation_ranks = rankdata([-relation_counter[relation] for relation in range(num_relations)])

    # Calculate probability distribution using ranks
    entity_probs = {entity: 1 / (entity_ranks[entity] + 1) for entity in range(num_entities)}
    relation_probs = {relation: 1 / (relation_ranks[relation] + 1) for relation in range(num_relations)}

    # Normalize probabilities
    entity_probs_sum = sum(entity_probs.values())
    relation_probs_sum = sum(relation_probs.values())
    entity_probs = {entity: prob / entity_probs_sum for entity, prob in entity_probs.items()}
    relation_probs = {relation: prob / relation_probs_sum for relation, prob in relation_probs.items()}

    # Sample entities/relations based on the probability distribution
    sampled_entities = np.random.choice(num_entities, size=batch_size, p=list(entity_probs.values()))
    sampled_relations = np.random.choice(num_relations, size=batch_size, p=list(relation_probs.values()))

    return [(entity, relation, entity) for entity, relation in zip(sampled_entities, sampled_relations)]


# Load and preprocess the knowledge graph
graph_file_path = "statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Generate negative examples

negative_triples = popularity_based_sampling(triples, num_entities, num_relations, batch_size=len(triples))
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
num_epochs = 5
batch_size = 256

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

# Create and compile the model
model = GAT_GCN_ComplexModelWithHoIE(num_entities, num_relations, embedding_dim, hoie_embedding_dim=embedding_dim)
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
mcc = matthews_corrcoef(y_true, y_pred)

# Display evaluation metrics
print("\nEvaluation Metrics:")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'PR AUC: {pr_auc:.4f}')
print(f'MCC: {mcc:.4f}')

# Plot training vs validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
