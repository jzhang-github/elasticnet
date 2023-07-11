
#import modules
import time
import tensorflow              as tf
import numpy                   as np
import os
import multiprocessing
import sklearn
from   sklearn.model_selection import KFold
import json

#import data, features and labels
def import_data(feature_file, label_file, SEED, drop_col:list=None):
    x_data = np.loadtxt(feature_file)
    y_data = np.loadtxt(label_file)

    if drop_col:
        all_cols = list(range(x_data.shape[1]))
        remain_cols = list(set(all_cols) - set(drop_col))
        x_data = x_data[:,remain_cols]

    #shuffle the data
    np.random.seed(SEED)
    np.random.shuffle(x_data)
    np.random.seed(SEED)
    np.random.shuffle(y_data)
    return x_data, y_data

def Training_module(kwargs):
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in kwargs['x_train_index']:
        x_train.append(kwargs['x_data'][i])
        y_train.append(kwargs['y_data'][i])
    for i in kwargs['x_test_index']:
        x_test.append(kwargs['x_data'][i])
        y_test.append(kwargs['y_data'][i])

    #convert the data into tf format
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test  = tf.convert_to_tensor(x_test,  dtype=tf.float32)
    y_test  = tf.convert_to_tensor(y_test,  dtype=tf.float32)

    #build an empty model
    model = tf.keras.models.Sequential()
    #add the input layer
    model.add(tf.keras.Input(shape=(x_train.shape[1],), name='Input_layer'))

    #add hidden layer(s)
    if kwargs['Regularization'] == True:
        print("L2 regularizers is used.")
        for i in range(len(kwargs['Nodes_per_layer'])):
            model.add(tf.keras.layers.Dense(kwargs['Nodes_per_layer'][i],
                                            activation=kwargs['Activation_function'],
                                            kernel_regularizer=tf.keras.regularizers.l2(),
                                            name="Hidden_layer_"+str(i)))
    elif kwargs['Regularization'] == False:
        for i in range(len(kwargs['Nodes_per_layer'])):
            model.add(tf.keras.layers.Dense(kwargs['Nodes_per_layer'][i],
                                            activation=kwargs['Activation_function'],
                                            name="Hidden_layer_"+str(i)))
    else:
        print("Illegal value for Regularization.")
        os._exit(0)

    # add the output layer
    if kwargs['Number_of_out_node'] == 'auto':
        outnodenum = np.shape(kwargs['y_data'])[1]
    else:
        outnodenum = kwargs['Number_of_out_node']
    model.add(tf.keras.layers.Dense(outnodenum,
                                    activation=kwargs['Output_activation'],
                                    name="Output_layer"))

    #compile
    model.compile(optimizer=kwargs['Optimizer'],
                  loss=kwargs['Cost_function'],
                   metrics=[kwargs['Metrics']])

    #fit the model
    history = model.fit(x_train, y_train,
                        batch_size=kwargs['Batch_size'], epochs=kwargs['Epochs'],
                        verbose=kwargs['Verbose'],
                        validation_data=(x_test, y_test), validation_freq=1)

    #print summary and trainable variables
    model.summary()
    file = open(f"{kwargs['Log_save_path']}/model_{kwargs['model_index']}_weights.txt", 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    #save the model
    model_name = f'{kwargs["Model_save_path"]}/model_{kwargs["model_index"]}_dense_layer.model'
    # model_name = os.path.join('checkpoint',
    #                           'cp.ckpt',
    #                           f'model_{kwargs["model_index"]}_dense_layer.model')
    model.save(model_name)

    #save history
    mae      = history.history['mean_absolute_error']
    val_mae  = history.history['val_mean_absolute_error']
    mape     = history.history['mean_absolute_percentage_error']
    val_mape = history.history['val_mean_absolute_percentage_error']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    #write result into disk
    acc_loss_filename = f'model_{kwargs["model_index"]}-{len(kwargs["Nodes_per_layer"])}_layer-{"".join([f"{x}_" for x in kwargs["Nodes_per_layer"]])}nodes.acc.loss'
    acc_loss_path     = os.path.join(kwargs['Log_save_path'], acc_loss_filename)
    with open(acc_loss_path, 'w') as f:
        f.write('epoch      train_mae        val_mae           mape       val_mape           loss       val_loss\n')
        out_log = np.vstack([history.epoch,
                             mae, val_mae, mape, val_mape, loss, val_loss]).T
        np.savetxt(f, out_log, fmt=['%5.0f', '%14.8f', '%14.8f', '%14.8f', '%14.8f', '%14.8f', '%14.8f'])

def CV_ML_RUN(config, drop_cols:list=None):
    now_time = time.time()
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        config = config

    kf = KFold(n_splits=config["Number_of_fold"])
    model_index = 0

    #create log file
    if not os.path.exists(config["Log_save_path"]):
        os.makedirs(config["Log_save_path"])

    print('Parent process %s.' % os.getpid())
    # x_data, y_data  = import_data(drop_cols)
    x_data, y_data  = import_data(config["feature_file"],
                                  config["label_file"],
                                  config["SEED"],
                                  drop_col=drop_cols)

    processes = []
    for x_train_index, x_test_index in kf.split(x_data):
        config['x_train_index'] = x_train_index
        config['x_test_index'] = x_test_index
        config['model_index'] = model_index
        p = multiprocessing.Process(target=Training_module, args=[config,])
        p.start()
        processes.append(p)
        model_index += 1
    print(processes)
    for process in processes:
        process.join()
    print('All subprocesses done.')

    # write the global result of every CV
    global_acc_loss_filename = f'{len(config["Nodes_per_layer"])}_layer-{"".join([f"{x}_" for x in config["Nodes_per_layer"]])}nodes.global.acc.loss'
    global_acc_loss_path     = os.path.join(config["Log_save_path"], global_acc_loss_filename)

    with open(global_acc_loss_path, 'w') as f:
        print('model                 mae       val_mae          mape      val_mape          loss      val_loss', file=f)
        mae_all, val_mae_all, mape_all, val_mape_all, loss_all, val_loss_all = [], [], [], [], [], []
        for i in range(config["Number_of_fold"]):
            fname = f'model_{i}-{len(config["Nodes_per_layer"])}_layer-{"".join([f"{x}_" for x in config["Nodes_per_layer"]])}nodes.acc.loss'
            with open(os.path.join(config["Log_save_path"], fname), 'r') as f_i:
                f_i.readline()
                data = np.loadtxt(f_i)
            epoch, mae, val_mae, mape, val_mape, loss, val_loss = data[-1]
            mae_all.append(mae)
            val_mae_all.append(val_mae)
            mape_all.append(mape)
            val_mape_all.append(val_mape)
            loss_all.append(loss)
            val_loss_all.append(val_loss)
            print('model {:<4.0f}  {:13.8f} {:13.8f} {:13.8f} {:13.8f} {:13.8f} {:13.8f}'.format(
                         i, mae, val_mae, mape, val_mape, loss, val_loss), file=f)
        print('===============================================================================================',
              file=f)
        print('mean        {:13.8f} {:13.8f} {:13.8f} {:13.8f} {:13.8f} {:13.8f}'.format(
                           np.mean(mae_all), np.mean(val_mae_all),
                           np.mean(mape_all), np.mean(val_mape_all),
                           np.mean(loss_all), np.mean(val_loss_all)), file=f)

    total_time = time.time() - now_time
    print("total_time", total_time, "s")

def load_and_pred(config, file_name_of_x_data, write_pred_log=True,
                  drop_cols:list=None):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        config = config
    #load new data
    x_pred = np.loadtxt(file_name_of_x_data)
    if drop_cols:
        all_cols = list(range(x_pred.shape[1]))
        remain_cols = list(set(all_cols) - set(drop_cols))
        x_pred = x_pred[:,remain_cols]

    #load the models and predict
    predictions_all = []
    for i in range(config["Number_of_fold"]):
        model_name = f'{config["Model_save_path"]}/model_{i}_dense_layer.model'
        new_model = tf.keras.models.load_model(model_name)
        predictions = new_model.predict([x_pred])
        predictions_all.append(predictions)
    predictions_all = np.array(predictions_all)
    prediction_mean = np.mean(predictions_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=1)

    if write_pred_log:
        if not os.path.exists(config["Prediction_save_path"]):
            os.makedirs(config["Prediction_save_path"])
        np.savetxt(str(config["Prediction_save_path"])+'/prediction_mean.txt',
                   prediction_mean, fmt='%16.8f')
        np.savetxt(str(config["Prediction_save_path"])+'/prediction_all.txt',
                   predictions_all, fmt='%16.8f')
    return prediction_mean, predictions_all
