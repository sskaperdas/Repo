import random

import numpy as np
import tensorflow as tf
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    matthews_corrcoef
from tensorflow.keras import layers
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


def combine_hole_embeddings(embed_real, embed_img):
    # Circular correlation for HolE
    combined_real = embed_real * embed_img
    combined_img = embed_real * embed_img
    return tf.concat([combined_real, combined_img], axis=-1)


class GCNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.activation_func = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]), self.units],
                                      initializer="glorot_uniform",
                                      trainable=True)
        self.bias = self.add_weight("bias",
                                    shape=[self.units],
                                    initializer="zeros",
                                    trainable=True)

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation_func is not None:
            output = self.activation_func(output)
        return output


# Define a more complex TransE link prediction model with a simple graph convolutional layer
class ComplexModelWithHoIE(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim, hoie_embedding_dim, gcn_units, dropout_rate=0.5, l2_reg=0.01):
        super(ComplexModelWithHoIE, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim, name='entity_embedding')
        self.hoie_embeddings = tf.keras.layers.Embedding(num_entities, hoie_embedding_dim * 2, name='hoie_embedding')
        self.relation_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim, name='relation_embedding')

        self.hoie_embedding_dim = hoie_embedding_dim

        self.hoie_dense1 = tf.keras.layers.Dense(hoie_embedding_dim, activation='relu')
        self.hoie_dense2 = tf.keras.layers.Dense(hoie_embedding_dim, activation='relu')

        self.gcn_layer1 = GCNLayer(gcn_units, activation='relu')
        self.gcn_layer2 = GCNLayer(gcn_units, activation='relu')

        self.conv1d_1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv1d_2 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

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

        hoie_embed_s = self.hoie_embeddings(s_idx)
        hoie_real_s, hoie_img_s = hoie_embed_s[:, :self.hoie_embedding_dim], hoie_embed_s[:, self.hoie_embedding_dim:]
        s_hoie = self.hoie_dense2(self.hoie_dense1(combine_hole_embeddings(hoie_real_s, hoie_img_s)))

        hoie_embed_o = self.hoie_embeddings(o_idx)
        hoie_real_o, hoie_img_o = hoie_embed_o[:, :self.hoie_embedding_dim], hoie_embed_o[:, self.hoie_embedding_dim:]
        o_hoie = self.hoie_dense2(self.hoie_dense1(combine_hole_embeddings(hoie_real_o, hoie_img_o)))

        concatenated_embed = tf.concat([s_embed, p_embed, o_embed, s_hoie, o_hoie], axis=1)

        # Pass the concatenated embeddings through the GCN layers
        gcn_output1 = self.gcn_layer1(concatenated_embed)
        gcn_output2 = self.gcn_layer2(gcn_output1)

        x = tf.expand_dims(gcn_output2, axis=-1)

        x = self.conv1d_1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.conv1d_2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        prediction = self.dense2(x)

        return prediction


# Function to flip entities in triples
def flip_entities(triples):
    flipped_triples = []
    for triple in triples:
        s, p, o = triple
        # Flip subject and object entities
        flipped_triples.append((o, p, s))
    return np.array(flipped_triples)


# Load and preprocess the knowledge graph
graph_file_path = "statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Generate negative examples

negative_triples = [(s, p, (o + np.random.randint(1, num_entities)) % num_entities) for s, p, o in triples]
all_triples = np.vstack((triples, np.array(negative_triples)))
labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

# Generate augmented data by flipping entities
flipped_triples = flip_entities(all_triples)

# Concatenate original and flipped triples
augmented_triples = np.vstack((all_triples, flipped_triples))

# Create set for initial positive triples for fast membership testing
positive_triples_set = set(map(tuple, triples))

# Assign labels to augmented triples based on membership in the initial positive triples set
augmented_labels = np.array([1 if tuple(triple) in positive_triples_set else 0 for triple in augmented_triples])

from sklearn.model_selection import StratifiedShuffleSplit

stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# Split augmented data into train, validation, and test sets
train_index, temp_index = next(stratified_splitter.split(augmented_triples, augmented_labels))
X_train_augmented, X_temp_augmented, y_train_augmented, y_temp_augmented = augmented_triples[train_index], augmented_triples[temp_index], augmented_labels[train_index], augmented_labels[temp_index]


stratified_splitter_test_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
test_index, val_index = next(stratified_splitter_test_val.split(X_temp_augmented, y_temp_augmented))
X_test_augmented, X_val_augmented, y_test_augmented, y_val_augmented = X_temp_augmented[test_index], X_temp_augmented[val_index], y_temp_augmented[test_index], y_temp_augmented[val_index]

# Create and compile the model
embedding_dim = 400
initial_learning_rate = 0.00005
num_epochs = 15
batch_size = 256

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

# Create and compile the model
model = ComplexModelWithHoIE(num_entities, num_relations, embedding_dim, embedding_dim, gcn_units=embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model on augmented data
history = model.fit(
    np.array(X_train_augmented), y_train_augmented,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(np.array(X_val_augmented), y_val_augmented),
    callbacks=[early_stopping, lr_schedule],
    verbose=1
)

# Evaluate the model on the test set
test_scores = model.predict(np.array(X_test_augmented))
y_pred = (test_scores > 0.5).astype(int)
y_true = y_test_augmented.astype(int)

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
