class Expert:
    """Minimal interface consumed by src.core.merge."""

    def get_layers(self):
        raise NotImplementedError

    def get_layer_params(self, layer_name):
        raise NotImplementedError

    def get_layer_cov(self, layer_name):
        return None

    def get_layer_fish(self, layer_name):
        return None

    def get_stat_fetcher_map(self, layer_name):
        return {
            "covariance": lambda: self.get_layer_cov(layer_name),
            "fisher": lambda: self.get_layer_fish(layer_name),
        }

    def get_layer_metadata(self, layer_name):
        return None

    def save_layer_params(self, tensor, layer_name, metadata=None):
        raise NotImplementedError

    def flush(self):
        pass

