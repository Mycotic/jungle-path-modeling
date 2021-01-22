class WindowGenerator():
    """
    This class is heavily based on the tensorflow guide. It allows you
    to create windows from a dataset. Currently a different one needs
    to be called for each window shape, which isn't ideal but is 
    complicated to improve.
    
        Arguments:
    input_width: int of how many time steps back the feature set should
                 include.
                 
    label_width: int of how many time steps the label set should
                 include. This is always 1 for windows meant for models,
                 but windows made for plotting have larger values.
                 
    shift:       Always should be 1 as far as I can tell, indicates
                 how far into the future to predict. 
                 
    eg_df:       Basically just used to get column names but ideally
                 should be removed.
    
    dfdi:        Dictionary of game dataframes, with gameids as index
    
    label_columns: List of the columns of the dfs which are the target.
    
    train_ids, val_ids, and test_ids: Lists of game ids determined earlier.
    
    
    
    """
    def __init__(self, input_width, label_width, shift,
                 eg_df, val_df,
                 dfdi,
                 train_ids,
                 val_ids,
                 test_ids,
                 label_columns=None):
        
        # Store the dfs etc in the class.
        self.eg_df = eg_df
        self.val_df = val_df
        self.dfdi = dfdi
        
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        
        self.trainli = None
        self.valli = None
        self.testli = None

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = ({name: i for i, name
                                          in enumerate(label_columns)})
        self.column_indices = {name: i for i, name in enumerate(eg_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])
    

    
    
    def split_window(self, features):
        """
        Splits window into labels and features.

        features: The full data being split. (tensor format I think)

        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            # Stacking the two labels because there's x and y!
            labels = tf.stack(
                    ([labels[:, :, self.column_indices[name]]
                      for name in self.label_columns]),
                    axis=-1)


        # Guide says this, not exactly sure what setting the shapes does exactly:
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the 'tf.data.Datasets' are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot_map2(self, model=None, input_width=3, plot_col=["playerx","playery"], max_subplots=3):
    """
    Plots predictions of 3 random windows from the val set of a given model.
    Returns None. Based in part on the tensorflow guide.
    
        Arguments:
    model: the model (tf objects only, probably).
    
    input_width: how many time steps the model predicts based on.
    
    plot_col: list of the two columns to predict, probably always the same.
    
    max_subplots: number of windows to plot.
    
    """
    
    inputs, labels = self.val
    # shuffle indices to get a random window, from
    # https://stackoverflow.com/questions/56575877/shuffling-two-tensors-in-the-same-order
    indices = tf.range(start=0, limit=tf.shape(inputs)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    inputs = tf.gather(inputs, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    
    plt.figure(figsize=(5, 15))
    # get label indices
    plot_col_index = [self.column_indices[plot_col[0]], self.column_indices[plot_col[1]]]
    # an easier way to grab label columns! very fashionable
    xy_inputs = tf.gather(inputs,plot_col_index,axis=2)
    
    # for loop, once for each window
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n+1)
        plt.xlim(0,15000)
        plt.ylim(0,15000)
        # plot the map background
        img = plt.imread("../data/map.png")
        plt.imshow(img, extent=[0,15000,0,15000],)
        
        # plot each line of the true data as well as labeling them with time
        for i in range(xy_inputs.shape[1]):
            plt.plot(xy_inputs[n, i:i+2, 0], xy_inputs[n, i:i+2, 1], zorder=10, color="blue")#color=(.8-i/10,1-i/8,1-i/8))
            plt.scatter(xy_inputs[n, i, 0], xy_inputs[n, i, 1],
                     marker='${}$'.format(str(i)), s=120, zorder=20, color="white")##color=(.8-i/10,1-i/8,1-i/8))
        
        # vestigial if
        if model is not None:
            # start by predicting each window  
            predictions = []
            for i in range(inputs.shape[1]-1):
                try:
                    current = model.predict(inputs[n,i:i+input_width,:])
                    predictions.append(current[0,:])
                except ValueError:
                    # an attempt to make plotmap work with multistep dense - didn't help
                    predictions = model.predict(tf.expand_dims(inputs[n,:,:],axis=0))
                    predictions = predictions[0,:,:]
                
            # plot line from each prediction to the last true value its based on
            # and keep the time label the same as the previous plotting
            for i in range(len(predictions)-1):
                line_x_0 = predictions[i][0]
                line_x_1 = predictions[i+1][0]
                line_y_0 = predictions[i][1]
                line_y_1 = predictions[i+1][1]
                plt.plot([xy_inputs[n, i+input_width-1, 0],line_x_0],
                         [xy_inputs[n, i+input_width-1, 1], line_y_0], zorder=10, color=(1,0,0))
                plt.scatter(line_x_0, line_y_0, marker='${}$'.format(str(i+input_width)),
                            s=120, zorder=20, color=(1,.8,.5))
        
        # plot the last label abusing i post for loop, very cool
        i=i+1
        plt.scatter(xy_inputs[n, i, 0], xy_inputs[n, i, 1],
                         marker='${}$'.format(str(i)), s=120, zorder=20, color="white")
        return None
    
    
    
    def make_dataset(self, data, list_me=False):
        """
        make_dataset converts a single game dataframe into windows.

            Arguments:
        data: Dataframe of a single game

        list_me: Whether to return a list of windows or a map of windows.
        """
        data = np.array(data, dtype=np.float32)

        # "Creates a dataset of sliding windows over a timeseries provided as array."
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,)

        ds = ds.map(self.split_window)
        if list_me:
            return list(ds)
        return ds



    def make_split_from_dfdi(self,minmax=(0,20)):
        """
        Combines the windows from each game generated by make_dataset.
            Arguments:
        minmax: Tuple of what timeframe to sample from (minutes).
        """
        # minmax is start and end time of window being sampled from.
        # splits is where the data is being split for train val test split
        if self.dfdi is None:
            print("no dfdi attached")
            raise KeyError("no dfdi attached")
        train_ids = self.train_ids
        val_ids = self.val_ids
        test_ids = self.test_ids
        if self.trainli is None:
            trainli = build_batch(self, self.dfdi, train_ids, minmax)
            train_feats = tf.concat([batch[0] for batch in trainli], axis=0)
            train_labels = tf.concat([batch[1] for batch in trainli], axis=0)

        if self.valli is None:
            valli = build_batch(self, self.dfdi, val_ids, minmax)
            val_feats = tf.concat([batch[0] for batch in valli], axis=0)
            val_labels = tf.concat([batch[1] for batch in valli], axis=0)

        if self.testli is None:
            testli = build_batch(self, self.dfdi, test_ids, minmax)
            test_feats = tf.concat([batch[0] for batch in testli], axis=0)
            test_labels = tf.concat([batch[1] for batch in testli], axis=0)


        print(train_feats.shape, train_labels.shape)
        self.train = (train_feats, train_labels)
        self.val = (val_feats, val_labels)
        self.test = (test_feats, test_labels)



    def build_batch(self, dfdi, id_li, minmax):
        """
        Calls make_dataset on a specific match and adds it to list. Probably
        doesn't make enough use of self.

            Arguments:
        dfdi: dictionary of shape matchid: dataframe.

        id_li: list of matchids to use.

        minmax: list of start and end time of what times to build from - if game is shorter than max, just uses last time
        returns a list of tf tensors.
        """
        dataset_li = []
        for match_id in id_li:
            df = dfdi[match_id]
            df = df.iloc[minmax[0]:minmax[1]]
            dataset_li = dataset_li + make_dataset(self, data=df, list_me=True)
        return dataset_li