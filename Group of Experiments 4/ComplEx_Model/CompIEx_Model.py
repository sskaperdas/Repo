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


class GCNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.kernel_real = self.add_weight("kernel_real", shape=[int(input_shape[-1]), self.units],
                                           initializer="glorot_uniform", trainable=True)
        self.kernel_imag = self.add_weight("kernel_imag", shape=[int(input_shape[-1]), self.units],
                                           initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)

        output_real = tf.matmul(inputs_real, self.kernel_real) - tf.matmul(inputs_imag, self.kernel_imag)
        output_imag = tf.matmul(inputs_real, self.kernel_imag) + tf.matmul(inputs_imag, self.kernel_real)

        output = tf.complex(output_real, output_imag)

        if self.activation is not None:
            output_real = self.activation(output_real)
            output_imag = self.activation(output_imag)
            output = tf.complex(output_real, output_imag)

        return output


class GCNComplEx_Model(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(GCNComplEx_Model, self).__init__()

        self.entity_real_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim,
                                                                name='entity_real_embedding')
        self.entity_imag_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim,
                                                                name='entity_imag_embedding')
        self.relation_real_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim,
                                                                  name='relation_real_embedding')
        self.relation_imag_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim,
                                                                  name='relation_imag_embedding')

        self.in_degree = tf.keras.layers.Embedding(num_entities, 1, name='in_degree_embedding')
        self.out_degree = tf.keras.layers.Embedding(num_entities, 1, name='out_degree_embedding')
        self.inverse_relation_frequency = tf.keras.layers.Embedding(num_relations, 1, name='inverse_relation_frequency')

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
        s_real_embed = self.entity_real_embeddings(inputs[:, 0])
        s_imag_embed = self.entity_imag_embeddings(inputs[:, 0])
        p_real_embed = self.relation_real_embeddings(inputs[:, 1])
        p_imag_embed = self.relation_imag_embeddings(inputs[:, 1])
        o_real_embed = self.entity_real_embeddings(inputs[:, 2])
        o_imag_embed = self.entity_imag_embeddings(inputs[:, 2])

        in_degree_embed = self.in_degree(inputs[:, 0])
        s_real_embed += in_degree_embed
        out_degree_embed = self.out_degree(inputs[:, 2])
        o_real_embed += out_degree_embed

        inverse_relation_frequency_embed = self.inverse_relation_frequency(inputs[:, 1])
        s_real_embed += inverse_relation_frequency_embed
        o_real_embed += inverse_relation_frequency_embed

        # Combine real and imaginary parts into complex numbers
        s_complex_embed = tf.complex(s_real_embed, s_imag_embed)
        p_complex_embed = tf.complex(p_real_embed, p_imag_embed)
        o_complex_embed = tf.complex(o_real_embed, o_imag_embed)

        # Concatenate embeddings for GCN input
        gcn_input = tf.concat([s_complex_embed, p_complex_embed, o_complex_embed], axis=-1)

        # GCN layers
        gcn_output_1 = self.gcn_layer_1(gcn_input)
        gcn_output_2 = self.gcn_layer_2(gcn_output_1)

        # Only take the real part for the dense layers
        concat_embed = tf.concat(
            [tf.math.real(s_complex_embed), tf.math.real(p_complex_embed), tf.math.real(o_complex_embed),
             tf.math.real(gcn_output_1), tf.math.real(gcn_output_2)], axis=1)

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

# Generate negative examples
negative_triples = [(s, p, (o + np.random.randint(1, num_entities)) % num_entities) for s, p, o in triples]
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

# Create and compile the GCNTransE model
embedding_dim = 256
initial_learning_rate = 0.0001
num_epochs = 15
batch_size = 256

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-7)

# Create and compile the model
model = GCNComplEx_Model(num_entities, num_relations, embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, lr_schedule], verbose=1)

# Evaluate the model on the test set
test_scores = model.predict(X_test)
y_pred = (test_scores > 0.5).astype(int)
y_true = y_test.astype(int)

# Calculate additional evaluation metrics
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
