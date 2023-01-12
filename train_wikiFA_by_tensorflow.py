import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main():
    # load training data
    X_train, y_train, X_test, y_test = load_dataset()

    # settings
    expt_name = "M1_dim100"
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    activation = 'linear'
    optimizer = 'adam'
    epochs = 100
    # dropout

    # Define the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(output_dim, input_shape={input_dim, }, use_bias=False, activation=activation))
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, workers=8, callbacks=[tf.keras.callbacks.History()])
    model.save(f'models/{optimizer}_{expt_name}.h5')

    # Predict using the model
    y_pred = model.predict(X_train)

    # Calculate cosine similarity between predicted and actual values
    cos_sim = np.array([cosine_similarity(y_pred[i], y_train[i]) for i in range(len(y_pred))])

    # 負の類似度を除外する
    cos_sim = cos_sim[cos_sim >= 0]

    print(f"sim average: {np.sum(cos_sim) / len(cos_sim)}")

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs[0].plot(history.history['loss'])
    axs[0].set_title(f'{expt_name} Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].plot(cos_sim)
    axs[1].set_title(f'{expt_name} Cosine Similarity')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Similarity')

    weights = model.get_weights()[0]
    axs[2].imshow(weights)
    axs[2].set_title(f'{expt_name} Weight Matrix')

    plt.show()

    print(model.summary())


def load_dataset():
    with open("dataset/wikiFA_docvecs_dim100.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset

# Calculate cosine similarity


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    main()
