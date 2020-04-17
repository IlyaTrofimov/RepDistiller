import torch.nn as nn

class DenseLayer(nn.Module):
    """DenseLayer
        Dense layer for NAS MNIST benchmark

    Linear module with dropout and ReLU activation.
    Used for builing models in NASModel.

    Args:
        drop_rate (float): dropout before rate.
        in_feature_num (int): number of input features.
        out_features_num (int): number of output features.
        isrelu (:obj:`bool`, optional): ReLU activation if True.
    Returns:
        torch.nn.module: linear layer with dropout
    """

    RESET_INIT_RANGE = 0.05

    def __init__(self,
                 drop_rate,
                 in_feature_num,
                 out_features_num,
                 isrelu):
        super(DenseLayer, self).__init__()

        self.drop_rate = drop_rate
        self.in_feature_num = in_feature_num
        self.out_features_num = out_features_num
        self.isrelu = isrelu

        if not (
                isinstance(self.drop_rate, float) and
                self.drop_rate <= 1.0 and
                self.drop_rate >= 0.0
                ):
            raise ValueError(
                "drop_rate is wrong: {}. "
                "Must be float from 0.0 to 1.0".format(self.drop_rate)
            )

        if not (
                isinstance(self.in_feature_num, int) and
                self.in_feature_num > 0
                ):
            raise ValueError(
                "in_feature_num is wrong: {}. "
                "Must be int more than 0".format(self.in_feature_num)
            )

        if not (
                isinstance(self.out_features_num, int) and
                self.out_features_num > 0
                ):
            raise ValueError(
                "out_features_num is wrong: {}. "
                "Must be int more than 0".format(self.out_features_num)
            )

        if not isinstance(self.isrelu, bool):
            raise ValueError("isrelu is wrong: {}. Must be bool".format(self.isrelu))

        if self.drop_rate > 0:
            self.dropout = nn.Dropout(self.drop_rate)
        self.linear = nn.Linear(self.in_feature_num, self.out_features_num)
        if self.isrelu:
            self.relu = nn.ReLU()

        self.reset_parameters()

    def forward(self, x):
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.linear(x)
        if self.isrelu:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-self.RESET_INIT_RANGE,
                                self.RESET_INIT_RANGE
                                )


class DenseModel(nn.Module):
    """DenseModel
        Dense model builder for NAS MNIST benchmark

    Args:
        config (:obj:`dict` of :obj:`list`): hidden_neurons and dropout_rates.
            hidden_neurons (:obj:`list` of :obj:`int`): number of hidden neurons on
                each layer.
            dropout_rates (:obj:`list` of :obj:`float`): dropout rate before each layer.
    Returns:
        torch.nn.module: NAS model
    """

    INPUT_FEATURES_NUM = 28*28

    def __init__(self, config):
        super(DenseModel, self).__init__()

        self.config = config
        if not isinstance(self.config, dict):
            raise TypeError(
                "config {} is wrong type {}. "
                "Can be only dict with hidden_neurons list and dropout_rates list"
                "".format(self.config, type(self.config))
            )
        if "hidden_neurons" not in self.config:
            raise Exception("hidden_neurons have to be in config")
        if "dropout_rates" not in self.config:
            raise Exception("dropout_rates have to be in config")
        if len(self.config['hidden_neurons']) != len(self.config['dropout_rates']):
            raise ValueError(
                "number of hidden_neurons "
                "must be the same "
                "as a dropout_rates configurations."
            )

        layers = nn.ModuleList([])
        cur_in_features_num = self.INPUT_FEATURES_NUM
        num_layers = len(self.config['hidden_neurons'])
        for idx in range(num_layers):
            isrelu = idx != (num_layers - 1)
            layers.append(
                DenseLayer(
                    self.config['dropout_rates'][idx],
                    cur_in_features_num,
                    self.config['hidden_neurons'][idx],
                    isrelu
                )
            )
            cur_in_features_num = self.config['hidden_neurons'][idx]

        #self.layers = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x, is_feat = False, preact = False):
        x = x.view(-1, self.INPUT_FEATURES_NUM)
        acts = []

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            acts.append(x)
        #x = self.layers(x)

        if is_feat:
            return acts, x
        else:
            return x
