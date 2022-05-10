"""
Configuration class for NN model structure.
"""

class Config():
    """
    This is a slimmed version of the pretrainedconfig from the Hugging Face repository.

    Args:
        n_ctx (int): Context window of transformer model.
        n_embd (int): Dimensionality of the embeddings.
        n_layer (int): Number of hidden layers in the transformer.
        n_head (int): Number of self-attention heads in each layer.
        state_dims (List): List of physical state dimensionality. Used in embedding models.
        activation_function (str, optional): Activation function. Defaults to "gelu_new".
        resid_pdrop (float, optional):
            The dropout probability for all fully connected layers in the transformer.
            Defaults to 0.0.
        embd_pdrop (float, optional):
            The dropout ratio for the embeddings. Defaults to 0.0.
        attn_pdrop (float, optional):
            The dropout ratio for the multi-head attention. Defaults to 0.0.
        layer_norm_epsilon (float, optional):
            The epsilon to use in the layer normalization layers. Defaults to 1e-5.
        initializer_range (float, optional):
            The standard deviation for initializing all weight matrices. Defaults to 0.02.
        output_hidden_states (bool, optional): Output embeddeding states from transformer. Defaults to False.
        output_attentions (bool, optional): Output attention values from transformer. Defaults to False.
        use_cache (bool, optional): Store transformers internal state for rapid predictions. Defaults to True.

    Raises:
        AssertionError: If provided parameter is not a config parameter
    """
    model_type: str = ""

    def __init__(self, **kwargs) -> None:

        # embedding NN
        self.n_embd = kwargs.pop("n_embd", 256) # for 10 pendula
        self.state_dims = kwargs.pop("state_dims", [20]) # for 10 pendula
        self.hidden_states = kwargs.pop("hidden_states", 1500) # for 10 pendula
        # self.n_embd = kwargs.pop("n_embd", 32) # for lorenz
        # self.state_dims = kwargs.pop("state_dims", [3]) # for lorenz
        # self.hidden_states = kwargs.pop("hidden_states", 500) # for lorenz

        self.layer_norm_epsilon = kwargs.pop("layer_norm_epsilon", 1e-5)
        
        # embedding NN loss term coefficients
        # original coeffs (lorenz)
        self.recons_lmbd = kwargs.pop("recons_lmbd", 1e4)
        self.dynamic_lmbd = kwargs.pop("dynamic_lmbd", 1e1)
        self.decay_lmbd = kwargs.pop("decay_lmbd", 1e-1)
        
        # Transformer NN
        self.n_ctx = kwargs.pop("n_ctx", 60) # for 10-pendula
        # self.n_ctx = kwargs.pop("n_ctx", 64) # for lorenz
        self.n_layer = kwargs.pop("n_layer", 4)
        self.n_head = kwargs.pop("n_head", 4)
        self.activation_function = kwargs.pop("activation_function", "gelu_new")
        self.initializer_range = kwargs.pop("initializer_range", 0.05)

        # Dropout regularization
        self.embd_pdrop = kwargs.pop("embd_pdrop", 0.0)
        self.resid_pdrop = kwargs.pop("resid_pdrop", 0.0)
        self.attn_pdrop = kwargs.pop("attn_pdrop", 0.0)
    
        # Output/Prediction related attributes
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_cache = kwargs.pop("use_cache", True)  # Not used by all models

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 8)
        self.min_length = kwargs.pop("min_length", 0)

        # Special parameters for different transformer models
        self.k_size = kwargs.pop("k_size", 1)