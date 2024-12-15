from random import random

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


# Define the holographic embeddings function
def combine_hole_embeddings(embed_real, embed_img):
    combined_real = embed_real * embed_img
    combined_img = embed_real * embed_img
    return tf.concat([combined_real, combined_img], axis=-1)


class GAT_ComplEx_Model_HoIE(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim, hoie_embedding_dim, num_heads, dropout_rate=0.3):
        super(GAT_ComplEx_Model_HoIE, self).__init__()

        self.entity_embeddings_real = tf.keras.layers.Embedding(
            num_entities, embedding_dim, name='entity_embedding_real')
        self.entity_embeddings_imag = tf.keras.layers.Embedding(
            num_entities, embedding_dim, name='entity_embedding_imag')

        self.relation_embeddings_real = tf.keras.layers.Embedding(
            num_relations, embedding_dim, name='relation_embedding_real')
        self.relation_embeddings_imag = tf.keras.layers.Embedding(
            num_relations, embedding_dim, name='relation_embedding_imag')

        self.hoie_embedding_dim = hoie_embedding_dim

        self.hoie_embeddings = tf.keras.layers.Embedding(
            num_entities, hoie_embedding_dim * 2, name='hoie_embedding')

        self.gat_layers = []
        for _ in range(1):
            self.gat_layers.append(GATLayer(num_heads=num_heads, output_dim=embedding_dim, dropout_rate=dropout_rate))

        self.dense1 = tf.keras.layers.Dense(
            5012, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(
            2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
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
        s_embed_real = self.entity_embeddings_real(inputs[:, 0])
        s_embed_imag = self.entity_embeddings_imag(inputs[:, 0])

        p_embed_real = self.relation_embeddings_real(inputs[:, 1])
        p_embed_imag = self.relation_embeddings_imag(inputs[:, 1])

        o_embed_real = self.entity_embeddings_real(inputs[:, 2])
        o_embed_imag = self.entity_embeddings_imag(inputs[:, 2])

        # Combine ComplEx embeddings
        s_embed = tf.concat([s_embed_real, s_embed_imag], axis=-1)
        p_embed = tf.concat([p_embed_real, p_embed_imag], axis=-1)
        o_embed = tf.concat([o_embed_real, o_embed_imag], axis=-1)

        # Retrieve holographic embeddings
        hoie_embed_s = self.hoie_embeddings(inputs[:, 0])
        hoie_real_s, hoie_img_s = hoie_embed_s[:, :self.hoie_embedding_dim], hoie_embed_s[:, self.hoie_embedding_dim:]
        s_hoie = combine_hole_embeddings(hoie_real_s, hoie_img_s)

        hoie_embed_o = self.hoie_embeddings(inputs[:, 2])
        hoie_real_o, hoie_img_o = hoie_embed_o[:, :self.hoie_embedding_dim], hoie_embed_o[:, self.hoie_embedding_dim:]
        o_hoie = combine_hole_embeddings(hoie_real_o, hoie_img_o)

        concat_embed = tf.concat([s_embed, p_embed, o_embed, s_hoie, o_hoie], axis=1)

        # Apply multiple GAT layers
        x = concat_embed
        for gat_layer in self.gat_layers:
            x = gat_layer(x)

        # Simplified graph convolution using dense layers
        x = self.dense1(x)
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

model = GAT_ComplEx_Model_HoIE(num_entities, num_relations, embedding_dim, num_heads=4, dropout_rate=0.5,
                               hoie_embedding_dim=embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss=MarginRankingLoss(margin=0.1),
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
    callbacks=[early_stopping, plateau_scheduler, tensorboard_callback],
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
