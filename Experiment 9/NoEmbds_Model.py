import random
import numpy as np
import tensorflow as tf
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    matthews_corrcoef
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers


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


class GCNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]), self.units],
                                      initializer="glorot_uniform",
                                      trainable=True)

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output


class GCN_NoEmbds_Model(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(GCN_NoEmbds_Model, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim, name='entity_embedding')
        self.relation_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim, name='relation_embedding')

        self.in_degree = tf.keras.layers.Embedding(num_entities, 1, name='in_degree_embedding')
        self.out_degree = tf.keras.layers.Embedding(num_entities, 1, name='out_degree_embedding')
        self.inverse_relation_frequency = tf.keras.layers.Embedding(num_relations, 1, name='inverse_relation_frequency')

        self.normalization = tf.keras.layers.BatchNormalization()

        # GCN layers
        self.gcn_layer_1 = GCNLayer(units=embedding_dim, activation=tf.nn.relu)
        self.gcn_layer_2 = GCNLayer(units=embedding_dim, activation=tf.nn.relu)

        self.dense1 = tf.keras.layers.Dense(5012, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(2048, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.dense3 = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout3 = tf.keras.layers.Dropout(0.5)

        self.dense4 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout4 = tf.keras.layers.Dropout(0.5)

        self.dense5 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout5 = tf.keras.layers.Dropout(0.5)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        s_embed = self.entity_embeddings(inputs[:, 0])
        p_embed = self.relation_embeddings(inputs[:, 1])
        o_embed = self.entity_embeddings(inputs[:, 2])

        in_degree_embed = self.in_degree(inputs[:, 0])
        s_embed += in_degree_embed

        out_degree_embed = self.out_degree(inputs[:, 2])
        o_embed += out_degree_embed

        inverse_relation_frequency_embed = self.inverse_relation_frequency(inputs[:, 1])
        s_embed += inverse_relation_frequency_embed
        o_embed += inverse_relation_frequency_embed

        s_embed = self.normalization(s_embed)
        o_embed = self.normalization(o_embed)

        # GCN layers
        gcn_output_1 = self.gcn_layer_1(tf.concat([s_embed, p_embed, o_embed], axis=1))
        gcn_output_2 = self.gcn_layer_2(gcn_output_1)

        concat_embed = tf.concat([s_embed, p_embed, o_embed, gcn_output_1, gcn_output_2], axis=1)

        x = self.dense1(concat_embed)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        skip_connection_1 = tf.keras.layers.Add()([s_embed, p_embed])
        skip_connection_2 = tf.keras.layers.Add()([o_embed, p_embed])

        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        x = self.dense4(x)
        x = self.dropout4(x, training=training)

        x = self.dense5(x)
        x = self.dropout5(x, training=training)

        flattened = self.flatten(x)
        prediction = self.dense(flattened)

        return prediction


# Function to flip entities in triples
def flip_entities(triples, num_entities):
    flipped_triples = []
    for triple in triples:
        s, p, o = triple
        # Flip subject and object entities
        flipped_triples.append((o, p, s))
    return np.array(flipped_triples)


# Load and preprocess the knowledge graph
graph_file_path = "temp/statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Generate negative examples

negative_triples = [(s, p, (o + np.random.randint(1, num_entities)) % num_entities) for s, p, o in triples]
all_triples = np.vstack((triples, np.array(negative_triples)))
labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

# Generate augmented data by flipping entities
flipped_triples = flip_entities(all_triples, num_entities)

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
X_train_augmented, X_temp_augmented, y_train_augmented, y_temp_augmented = augmented_triples[train_index], \
augmented_triples[temp_index], augmented_labels[train_index], augmented_labels[temp_index]

stratified_splitter_test_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
test_index, val_index = next(stratified_splitter_test_val.split(X_temp_augmented, y_temp_augmented))
X_test_augmented, X_val_augmented, y_test_augmented, y_val_augmented = X_temp_augmented[test_index], X_temp_augmented[
    val_index], y_temp_augmented[test_index], y_temp_augmented[val_index]

# Create and compile the model
embedding_dim = 256
initial_learning_rate = 0.0001
num_epochs = 15
batch_size = 256

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-7)

# Create and compile the model with Margin Ranking Loss
model = GCN_NoEmbds_Model(num_entities, num_relations, embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss=MarginRankingLoss(margin=0.1), metrics=['accuracy'])

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
