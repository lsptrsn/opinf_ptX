from dataclasses import dataclass, field
import numpy as np


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Params(metaclass=Singleton):
    # input data
    data_dir: str = 'data/exp_data'
    file_suffix: str = '_dyn_T_ramps_exp'
    split: str = 'time' # 'time', 'condition'

    hidden_states: bool = True
    artificial_noise: bool = False
    # sample data size
    step_t_sampling: int = 10
    step_z_sampling: int = 1
    time_to_leave_out: int = 100

    # training split
    train_ratio: float = 0.7

    # data processing
    reduced_scaling: bool = True
    true_derivatives: bool = False
    state_scaling: bool = 'simple' # min-max, simple, none
    state_shifting: str ='mean' # max, mean, none, steady-state
    input_scaling: bool = True

    # modele
    model_structure: str = 'AHB'   # 'A', 'H', 'C', 'B'
    stability: str = 'global'  # local, global, none
    local_set_Q_to_identity: bool = False
    basis: str = 'NL-POD'  # POD, NL-POD, AM
    regularization_H: float = 0.0003958345668757349 # regularization
    regularization_A: float = 1e-6

    # dealing with hidden states
    lag_max: int = 3 # maximum of allowed lags (non-markov)
    completion: str = 'linear' # 'none', 'linear', 'knowledge-based'

    # dealing with noise
    noise_level: float = 0
    use_filter: str = True
    filter_window = 309
    filter_poly = 5
    ddt_method: str = 'savgol' #'whittaker' #'spline_cv', 'savgol', 'ord6', 'ms', 'composite', 'whittaker'
    ddt_postprocessing: bool = True  # apply savgol filter after getting ddt
    ROM_loss: str = 'hybrid' # 'ddt', states, # 'hybrid'
    lambda_deriv: float = 0.4219487078563753
    lambda_state: float = 1-lambda_deriv

    # pinn parameters
    use_PINN: bool = False
    PINN_state_smoothing: bool = False
    PINN_loss: str = 'mse´' # mse or huber
    PINN_lr_net: float = 2e-4
    PINN_lr_opinf: float = 2e-3
    PINN_architecture: str = "feedforward"   # "feedforward" or "siren"
    PINN_epochs: int = 1000
    PINN_batch_size: int = 2703
    PINN_phys_weight: int = 1
    PINN_activation: str = 'sine'  # selu or sine or tanh
    PINN_hidden_layers: list = (256, 256, 256, 256)
    PINN_stage_config_A: float = 0.4
    PINN_stage_config_B: float = 0.2

    # lifting
    apply_lifting_1: bool = False
    apply_lifting_2: bool = False

    # ROM size
    tolerance: float = 1e-3
    energy_threshold_single: float = 0.9922563283170605
    thresholds: np.ndarray = field(
        default_factory=lambda: np.arange(0.99, 0.99991, 1))
    r_F: int = 2  # basis size for conversion
    r_T: int = 6  # basis size for temperature
    ROM_order: int = 8  # summed up order of the reduced system
    input_dim: int = 3  # dimension of the control

    # saving and plotting
    save_results: bool = False
    output: bool = False

    # fitting
    batch_size = 1000000 # Full batch
    num_epochs = 1000

    # adam
    adam_lr: float = 0.04065651701003823
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0

    # lr_schedule
    lr_schedule_step_factor: float = 1725
    lr_schedule_mode: str = "triangular2"
    lr_schedule_cycle_momentum: bool = False
    lr_schedule_base_lr: float = adam_lr / 5.0
    lr_schedule_max_lr: float = adam_lr

    # CNN
    use_CNN: bool = True
    single_CNN: bool = True
    CNN_epochs: int = 1000 # dont change
    CNN_batch_size: int = 64 # dont change
    CNN_learn_rate: float = 0.00047445581557196064
    CNN_weight_decay: float = 1e-5
    CNN_dropout_rate_lin: float = 0.0
    CNN_dropout_rate_conv: float = 0.0
    CNN_loss_function: str = 'mse'  # 'mse', 'mae', 'mse_mae', 'smooth'
    CNN_gradient_clipping: bool = True
    CNN_activation = 'gelu' # "relu", "leakyrelu", "elu", "selu", "silu", "gelu", "softplus"
    CNN_patience: int = 500
    CNN_max_grad_norm: float = 1
    CNN_input_noise: float = 0.009949372596871502
    CNN_conv_channel: tuple=(64, 64, 16)
