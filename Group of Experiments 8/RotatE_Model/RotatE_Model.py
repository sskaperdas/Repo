import numpy as np
import tensorflow as tf
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    matthews_corrcoef
from tensorflow.keras.callbacks import EarlyStopping


tf.random.set_seed(42)
np.random.seed(42)


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


# Define the triple classification RotatE model with the Conv1D architecture
class GCN_Complex_Model_With_RotatE(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim, rot_embedding_dim, dropout_rate=0.5, l2_reg=0.01):
        super(GCN_Complex_Model_With_RotatE, self).__init__()

        self.entity_embeddings = tf.keras.layers.Embedding(num_entities, embedding_dim * 2, name='entity_embedding')
        self.relation_embeddings = tf.keras.layers.Embedding(num_relations, embedding_dim, name='relation_embedding')

        # Additional RotatE embeddings
        self.rot_entity_embeddings = tf.keras.layers.Embedding(num_entities, rot_embedding_dim * 2,
                                                               name='rot_entity_embedding')
        self.rot_relation_embeddings = tf.keras.layers.Embedding(num_relations, rot_embedding_dim,
                                                                 name='rot_relation_embedding')

        # Graph Convolution Layers
        self.conv1d_1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv1d_2 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # GCN Layers
        self.gcn_layer1 = GCNLayer(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.gcn_layer2 = GCNLayer(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, inputs, training=None, mask=None):
        s_idx, p_idx, o_idx = inputs[:, 0], inputs[:, 1], inputs[:, 2]

        # RotatE embeddings
        rot_s_embed = self.rot_entity_embeddings(s_idx)
        rot_p_embed = self.rot_relation_embeddings(p_idx)
        rot_o_embed = self.rot_entity_embeddings(o_idx)

        # Splitting RotatE embeddings into real and imaginary parts
        s_real, s_imag = tf.split(rot_s_embed, 2, axis=-1)
        o_real, o_imag = tf.split(rot_o_embed, 2, axis=-1)

        # Rotation
        re_score = tf.reduce_sum(rot_p_embed * ((s_real * o_real) + (s_imag * o_imag)), axis=-1)
        im_score = tf.reduce_sum(rot_p_embed * ((s_imag * o_real) - (s_real * o_imag)), axis=-1)

        # Combining real and imaginary scores
        scores = tf.stack([re_score, im_score], axis=-1)
        scores = tf.reduce_sum(scores, axis=-1)
        x = tf.expand_dims(scores, axis=-1)  # Add channel dimension for Conv1D

        # Convolutional layers
        x = self.conv1d_1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.conv1d_2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        # GCN Layers
        x = self.gcn_layer1(x)
        x = self.gcn_layer2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        # Output layer
        prediction = self.dense2(x)

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

# Create and compile the model
embedding_dim = 400
initial_learning_rate = 0.00005
num_epochs = 15
batch_size = 256

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

# Create and compile the model
model = GCN_Complex_Model_With_RotatE(num_entities, num_relations, embedding_dim, embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Reshape input data to have three dimensions
X_train_reshaped = X_train.reshape(-1, 3, 1)
X_val_reshaped = X_val.reshape(-1, 3, 1)
X_test_reshaped = X_test.reshape(-1, 3, 1)

# Train the model with reshaped input data
history = model.fit(
    X_train_reshaped, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_val_reshaped, y_val),
    callbacks=[early_stopping, lr_schedule],
    verbose=1
)


# Evaluate the model on the test set
test_scores = model.predict(np.array(X_test_reshaped))
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
