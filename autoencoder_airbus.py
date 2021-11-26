def auto_encoder_1(vectorization_equal_duration_1):

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pylab import rcParams
    import random


    import tensorflow as tf
    from keras import optimizers, Sequential
    from keras.models import Model
    from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
    from keras.callbacks import ModelCheckpoint, TensorBoard

    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    #from sklearn.metrics import confusion_matrix, precision_recall_curve
    #from sklearn.metrics import recall_score, classification_report, auc, roc_curve
    #from sklearn.metrics import precision_recall_fscore_support, f1_score

    from numpy.random import seed
    seed(7)
    from tensorflow import set_random_seed
    set_random_seed(11)
    #from scipy.spatial.distance import cdist

    SEED = 123 #used to help randomly select the data points
    DATA_SPLIT_PCT = 0.2

    rcParams['figure.figsize'] = 8, 6
    LABELS = ["Normal","Break"]

    df = vectorization_equal_duration_1
    size_timeseries = [int(x.len_seq_ff1) for x in df.select("len_seq_ff1").collect()][0]
    L_label_id = [str(x.label_id) for x in df.select("label_id").collect()]
    L_timeseries = [x.ff1_seq for x in df.select("ff1_seq").collect()]

    l = []
    for elements in L_timeseries:
        for e in elements:
            l.append(e)

    input_X = np.array([l])
    input_X = input_X.T
    n_features = input_X.shape[1]

    min_max_scaler = preprocessing.StandardScaler()  #MinMaxScaler() pas trop significatif
    data_normalized = min_max_scaler.fit_transform(input_X)
    #tf.keras.utils.normalize(data)

    print(data_normalized[0])

    input_X_scaled = data_normalized.reshape((-1,size_timeseries, n_features)) 
    print(input_X_scaled.shape)

    X_train_scaled, X_test_scaled = train_test_split(np.array(input_X_scaled), test_size=DATA_SPLIT_PCT, random_state=SEED)
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)
    X_train_scaled, X_valid_scaled = train_test_split(X_train_scaled, test_size=DATA_SPLIT_PCT, random_state=SEED)
    print(X_train_scaled.shape)
    print(X_valid_scaled.shape)

    # to be removed
    #X_train_scaled = X_train_scaled[1:1000]

    timesteps =  X_train_scaled.shape[1] 
    n_features =  X_train_scaled.shape[2]

    epochs = 10 # 450 #200
    batch = 20 # 40
    lr = 0.01 #0.0001

    lstm_autoencoder_1 = Sequential()
    # Encoder
    lstm_autoencoder_1.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    lstm_autoencoder_1.add(LSTM(16, activation='relu', return_sequences=False))
    lstm_autoencoder_1.add(RepeatVector(timesteps))
    # Decoder
    lstm_autoencoder_1.add(LSTM(16, activation='relu', return_sequences=True))
    lstm_autoencoder_1.add(LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder_1.add(TimeDistributed(Dense(n_features)))

    lstm_autoencoder_1.summary()

    adam = optimizers.Adam(lr)
    lstm_autoencoder_1.compile(loss='mae', optimizer=adam)

    cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                                save_best_only=True,
                                verbose=0)

    tb = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)

    lstm_autoencoder_1_history = lstm_autoencoder_1.fit(X_train_scaled, X_train_scaled, 
                                                    epochs=epochs, 
                                                    batch_size=batch, 
                                                    validation_data=(X_valid_scaled, X_valid_scaled),
                                                    verbose=2).history

    cpt = 0
    cpt_anomalies = 0

    record_nb_to_display = random.randint(0,len(X_test_scaled))
    to_predict = X_test_scaled[record_nb_to_display]
    predict = lstm_autoencoder_1.predict(to_predict.reshape(1, 60, 1))[0]
    plt.plot(to_predict)
    plt.plot(predict)
    plt.title("Visualization of ff1")
    plt.xlabel('normalized time')
    plt.ylabel('ff1 value')
    plt.plot(to_predict, linewidth=2, label='Target')
    plt.plot(predict, linewidth=2, label='prediction')
    plt.legend(loc='upper right')
    plt.show()
    # print("mean = " + str(np.mean((to_predict.reshape(60)-predict.reshape(60))**2)))
    cost = np.sum((to_predict.reshape(60)-predict.reshape(60))**2)
    print("cost = " + str(cost))
    if cost > 25:
        print("Anomaly detected")
        cpt_anomalies = cpt_anomalies + 1
    else:
        print("No Anomaly detected")
    print("\n(record_displayed : " + str(record_nb_to_display) + ")")
    cpt = cpt+1

    tf.keras.backend.clear_session()
