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
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), workers=8, callbacks=[tf.keras.callbacks.History()])
    model.save(f'models/{expt_name}_{optimizer}.h5')

    # Predict using the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate cosine similarity between predicted and actual values
    cos_sim_train = np.array([cosine_similarity(y_pred_train[i], y_train[i]) for i in range(len(y_pred_train))])
    cos_sim_test = np.array([cosine_similarity(y_pred_test[i], y_test[i]) for i in range(len(y_pred_test))])

    print(f"sim average train: {np.sum(cos_sim_train) / len(cos_sim_train)}")
    print(f"sim average test: {np.sum(cos_sim_test) / len(cos_sim_test)}")

    # グラフを描画
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs[0].plot(history.history['loss'], label='train_loss')
    axs[0].plot(history.history['val_loss'], label='test_loss')
    axs[0].set_title(f'Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(cos_sim_train, label='sim_train')
    axs[1].plot(cos_sim_test, label='sim_test')
    axs[1].set_title(f'Cosine Similarity')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Similarity')

    # weights = model.get_weights()[0]
    # axs[2].imshow(weights)
    # axs[2].set_title(f'Weight Matrix')

    fig.suptitle(f'{expt_name}_{optimizer} Results')
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
