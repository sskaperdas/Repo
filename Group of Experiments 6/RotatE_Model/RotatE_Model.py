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

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)


# Load and preprocess the knowledge graph from TTL file
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


class ComplexMarginRankingLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.1):
        super(ComplexMarginRankingLoss, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        # Separate real and imaginary parts of predictions
        y_pred_real = tf.math.real(y_pred)
        y_pred_imag = tf.math.imag(y_pred)

        # Extract positive and negative scores for real and imaginary parts
        pos_scores_real = tf.boolean_mask(y_pred_real, y_true == 1)
        pos_scores_imag = tf.boolean_mask(y_pred_imag, y_true == 1)
        neg_scores_real = tf.boolean_mask(y_pred_real, y_true == 0)
        neg_scores_imag = tf.boolean_mask(y_pred_imag, y_true == 0)

        # Expand dimensions for broadcasting
        pos_scores_real = tf.expand_dims(pos_scores_real, axis=0)
        pos_scores_imag = tf.expand_dims(pos_scores_imag, axis=0)
        neg_scores_real = tf.expand_dims(neg_scores_real, axis=1)
        neg_scores_imag = tf.expand_dims(neg_scores_imag, axis=1)

        # Compute the margin-based loss
        margin_matrix_real = self.margin - pos_scores_real + neg_scores_real
        margin_matrix_imag = self.margin - pos_scores_imag + neg_scores_imag
        loss_real = tf.reduce_sum(tf.maximum(0.0, margin_matrix_real))
        loss_imag = tf.reduce_sum(tf.maximum(0.0, margin_matrix_imag))

        # Combine real and imaginary parts for the final loss
        loss = loss_real + loss_imag

        return loss


class ComplexGATLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, output_dim, activation='relu', dropout_rate=0.3):
        super(ComplexGATLayer, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernels_real = [self.add_weight('kernel_head_{}_real'.format(i),
                                             shape=(input_dim, self.output_dim),
                                             initializer='glorot_uniform',
                                             trainable=True)
                             for i in range(self.num_heads)]
        self.kernels_imag = [self.add_weight('kernel_head_{}_imag'.format(i),
                                             shape=(input_dim, self.output_dim),
                                             initializer='glorot_uniform',
                                             trainable=True)
                             for i in range(self.num_heads)]
        self.biases_real = [self.add_weight('bias_head_{}_real'.format(i),
                                            shape=(self.output_dim,),
                                            initializer='zeros',
                                            trainable=True)
                            for i in range(self.num_heads)]
        self.biases_imag = [self.add_weight('bias_head_{}_imag'.format(i),
                                            shape=(self.output_dim,),
                                            initializer='zeros',
                                            trainable=True)
                            for i in range(self.num_heads)]

    def call(self, inputs, training=None):
        outputs_real = []
        outputs_imag = []

        for i in range(self.num_heads):
            kernel_real, kernel_imag = self.kernels_real[i], self.kernels_imag[i]
            bias_real, bias_imag = self.biases_real[i], self.biases_imag[i]
            input_real, input_imag = tf.math.real(inputs), tf.math.imag(inputs)

            # Real part computation
            output_real = tf.matmul(input_real, kernel_real) - tf.matmul(input_imag, kernel_imag)
            output_real += bias_real

            # Imaginary part computation
            output_imag = tf.matmul(input_real, kernel_imag) + tf.matmul(input_imag, kernel_real)
            output_imag += bias_imag

            outputs_real.append(output_real)
            outputs_imag.append(output_imag)

        # Combine real and imaginary parts
        outputs_real = tf.concat(outputs_real, axis=-1)
        outputs_imag = tf.concat(outputs_imag, axis=-1)

        # Apply activation function
        outputs_real = self.activation(outputs_real)
        outputs_imag = self.activation(outputs_imag)

        # Apply dropout
        outputs_real = tf.keras.layers.Dropout(self.dropout_rate)(outputs_real, training=training)
        outputs_imag = tf.keras.layers.Dropout(self.dropout_rate)(outputs_imag, training=training)

        # Combine real and imaginary parts into complex tensor
        outputs = tf.complex(outputs_real, outputs_imag)
        return outputs


class ComplexGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(ComplexGCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.activation_func = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel_real = self.add_weight("kernel_real",
                                           shape=[input_dim, self.units],
                                           initializer="glorot_uniform",
                                           trainable=True)
        self.kernel_imag = self.add_weight("kernel_imag",
                                           shape=[input_dim, self.units],
                                           initializer="glorot_uniform",
                                           trainable=True)
        self.bias_real = self.add_weight("bias_real",
                                         shape=[self.units],
                                         initializer="zeros",
                                         trainable=True)
        self.bias_imag = self.add_weight("bias_imag",
                                         shape=[self.units],
                                         initializer="zeros",
                                         trainable=True)
        self.kernel = tf.complex(self.kernel_real, self.kernel_imag)  # Combine real and imaginary parts

    def call(self, inputs):
        input_real, input_imag = tf.math.real(inputs), tf.math.imag(inputs)
        output_real = tf.matmul(input_real, self.kernel_real) - tf.matmul(input_imag, self.kernel_imag)
        output_imag = tf.matmul(input_real, self.kernel_imag) + tf.matmul(input_imag, self.kernel_real)
        output_real += self.bias_real
        output_imag += self.bias_imag
        if self.activation_func is not None:
            output_real = self.activation_func(output_real)
            output_imag = self.activation_func(output_imag)
        return tf.complex(output_real, output_imag)


class RotatEEncoder(tf.keras.layers.Layer):
    def __init__(self, num_entities, num_relations, embedding_dim, activation=None):
        super(RotatEEncoder, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.activation = activation

    def build(self, input_shape):
        self.entity_embedding_real = self.add_weight("entity_embedding_real",
                                                     shape=(self.num_entities, self.embedding_dim),
                                                     initializer="glorot_uniform",
                                                     trainable=True)
        self.entity_embedding_img = self.add_weight("entity_embedding_img",
                                                    shape=(self.num_entities, self.embedding_dim),
                                                    initializer="glorot_uniform",
                                                    trainable=True)
        self.relation_embedding_real = self.add_weight("relation_embedding_real",
                                                       shape=(self.num_relations, self.embedding_dim),
                                                       initializer="glorot_uniform",
                                                       trainable=True)
        self.relation_embedding_img = self.add_weight("relation_embedding_img",
                                                      shape=(self.num_relations, self.embedding_dim),
                                                      initializer="glorot_uniform",
                                                      trainable=True)

    def call(self, inputs):
        s, p, o = inputs[:, 0], inputs[:, 1], inputs[:, 2]

        s_embed_real = tf.nn.embedding_lookup(self.entity_embedding_real, s)
        s_embed_img = tf.nn.embedding_lookup(self.entity_embedding_img, s)

        p_embed_real = tf.nn.embedding_lookup(self.relation_embedding_real, p)
        p_embed_img = tf.nn.embedding_lookup(self.relation_embedding_img, p)

        o_embed_real = tf.nn.embedding_lookup(self.entity_embedding_real, o)
        o_embed_img = tf.nn.embedding_lookup(self.entity_embedding_img, o)

        s_embed = tf.complex(s_embed_real, s_embed_img)
        p_embed = tf.complex(p_embed_real, p_embed_img)
        o_embed = tf.complex(o_embed_real, o_embed_img)

        return s_embed, p_embed, o_embed


class RotatEDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(RotatEDecoder, self).__init__()

    def call(self, inputs):
        s_embed, p_embed, o_embed = inputs

        score = tf.reduce_sum(s_embed * p_embed * tf.math.conj(o_embed), axis=-1)
        score = tf.math.angle(score)

        return score


class GCN_GAT_RotatE_Model(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim, num_heads, dropout_rate=0.3):
        super(GCN_GAT_RotatE_Model, self).__init__()

        self.encoder = RotatEEncoder(num_entities, num_relations, embedding_dim)
        self.decoder = RotatEDecoder()

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.gcn_layers = []
        for _ in range(2):
            self.gcn_layers.append(ComplexGCNLayer(units=embedding_dim, activation='relu'))

        self.gat_layers = []
        for _ in range(2):
            self.gat_layers.append(ComplexGATLayer(num_heads=num_heads, output_dim=embedding_dim, dropout_rate=dropout_rate))

        self.dense_layers = [
            tf.keras.layers.Dense(5012, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ]

        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        s_embed, p_embed, o_embed = self.encoder(inputs)

        # Separate real and imaginary parts
        s_embed_real, s_embed_img = tf.math.real(s_embed), tf.math.imag(s_embed)
        p_embed_real, p_embed_img = tf.math.real(p_embed), tf.math.imag(p_embed)
        o_embed_real, o_embed_img = tf.math.real(o_embed), tf.math.imag(o_embed)

        # Apply dropout separately to real and imaginary parts
        s_embed_real = self.dropout(s_embed_real, training=training)
        s_embed_img = self.dropout(s_embed_img, training=training)
        p_embed_real = self.dropout(p_embed_real, training=training)
        p_embed_img = self.dropout(p_embed_img, training=training)
        o_embed_real = self.dropout(o_embed_real, training=training)
        o_embed_img = self.dropout(o_embed_img, training=training)

        # Combine real and imaginary parts back into complex tensors
        s_embed = tf.complex(s_embed_real, s_embed_img)
        p_embed = tf.complex(p_embed_real, p_embed_img)
        o_embed = tf.complex(o_embed_real, o_embed_img)

        # Calculate the relation-specific transformed embeddings
        transformed_o_embed = s_embed * p_embed

        # Concatenate the embeddings
        concat_embed = tf.concat([s_embed, transformed_o_embed, o_embed], axis=-1)

        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            concat_embed = gcn_layer(concat_embed)

        # Apply GAT layers
        for gat_layer in self.gat_layers:
            concat_embed = gat_layer(concat_embed)

        # Apply dense layers
        for dense_layer in self.dense_layers:
            concat_embed = dense_layer(concat_embed)

        # Final dense layer for prediction
        prediction = self.final_dense(concat_embed)

        return prediction


# Load and preprocess the knowledge graph
graph_file_path = "statements.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Compute entity degrees
entity_degrees = compute_entity_degrees(triples)

# Generate negative examples

negative_triples = [(s, p, (o + np.random.randint(1, num_entities)) % num_entities) for s, p, o in triples]
all_triples = np.vstack((triples, negative_triples))
labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

# Shuffle and split data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(all_triples, labels, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and compile the model
embedding_dim = 512
num_epochs = 50
batch_size = 256

model = GCN_GAT_RotatE_Model(num_entities, num_relations, embedding_dim, num_heads=4, dropout_rate=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss=ComplexMarginRankingLoss(margin=0.1),
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
