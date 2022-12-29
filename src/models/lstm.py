import math
from typing import Any, Tuple, TypeVar

import torch

from src.utils import types

LSTMStatesType = TypeVar("LSTMStatesType", bound=Tuple[torch.Tensor, torch.Tensor])
LSTMOutputType = TypeVar("LSTMOutputType", torch.Tensor, Tuple[torch.Tensor, torch.Tensor])


class AbstractLSTMOperator(torch.nn.Module):
    def __init__(self, n_input_state_variables: int, n_output_state_variables: int):
        """
        :param n_input_state_variables: number of state input variables, equivalent to LSTM input size
        :param n_output_state_variables: number of states to produce, equivalent to LSTM hidden state size
        """
        super().__init__()

        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Can't call AbstractLSTMOperator!")


class ClassicLSTMOperator(AbstractLSTMOperator):
    """
    Operator implemented in standard LSTM network
    Processes each input and hidden state by applying formula:
        x @ u + h @ v + b
    Where:
        * `x` - inputs
        * `h` - network hidden state
        * `u` - neural network matrix parameter with shape (I, H)
        * `v` - neural network matrix parameter with shape (H, H)
        * `b` - neural network vector parameter (bias vector) with length H
    """

    def __init__(self, n_input_state_variables: int, n_output_state_variables: int, use_bias: bool = True):
        """
        :param n_input_state_variables: number of state input variables, equivalent to LSTM input size
        :param n_output_state_variables: number of states to produce, equivalent to LSTM hidden state size
        :param use_bias: if True bias vector is additional network learned parameter
        """
        super().__init__(n_input_state_variables, n_output_state_variables)

        self.u = torch.nn.Parameter(torch.Tensor(n_input_state_variables, n_output_state_variables))
        self.v = torch.nn.Parameter(torch.Tensor(n_output_state_variables, n_output_state_variables))
        self.b = torch.nn.Parameter(torch.Tensor(n_output_state_variables)) if use_bias else 0.0

        self.initialized = False

    def initialize(self):
        """Initialize parameter values using default pytorch method for LSTM"""
        with torch.no_grad():
            deviation = 1.0 / math.sqrt(self.n_output_state_variables)
            for parameter in self.parameters():
                parameter.data.uniform_(-deviation, deviation)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.initialize()

        return inputs @ self.u + hidden @ self.v + self.b


class DeepLSTMOperator(AbstractLSTMOperator):
    """
    Operator implemented in standard LSTM network
    Processes each input and hidden state by applying formula:
        f(x) + g(h) + b
    Where:
        * `x` - inputs
        * `h` - network hidden state
        * `f` - neural network with linear operators and non-linear activations processing inputs
        * `g` - neural network with linear operators and non-linear activations processing hidden states
        * `b` - neural network vector parameter (bias vector) with length H
    """

    def __init__(
        self,
        n_input_state_variables: int,
        n_output_state_variables: int,
        input_network: torch.nn.Module,
        state_network: torch.nn.Module,
        use_bias: bool = True,
    ):
        """
        :param n_input_state_variables: number of state input variables, equivalent to LSTM input size
        :param n_output_state_variables: number of states to produce, equivalent to LSTM hidden state size
        :param input_network: neural network with linear operators and non-linear activations processing inputs
        :param state_network: neural network with linear operators and non-linear activations processing hidden states
        :param use_bias: if True bias vector is additional network learned parameter
        """
        super().__init__(n_input_state_variables, n_output_state_variables)

        self.input_network = input_network
        self.state_network = state_network
        self.bias = torch.nn.Parameter(torch.Tensor(n_output_state_variables)) if use_bias else 0.0

        self.initialized = False

    @classmethod
    def from_network_parameters(
        cls,
        n_input_state_variables: int,
        n_output_state_variables: int,
        input_network_depth: int,
        state_network_depth: int,
        input_network_hidden_state_variables: int,
        state_network_hidden_state_variables: int,
        input_network_activation: callable,
        state_network_activation: callable,
        use_bias: bool = True,
    ):
        """
        :param n_input_state_variables: number of state input variables, equivalent to LSTM input size
        :param n_output_state_variables: number of states to produce, equivalent to LSTM hidden state size
        :param input_network_depth: number of linear layers used in input network
                                    1 results in 2 layer network transforming from input dimension to given hidden
                                    dimension and to output dimension, which is required to keep consistent with LSTM
        :param state_network_depth: number of linear layers used in state network applied analogously to input_network
        :param input_network_hidden_state_variables: number of dimensions in hidden representation of input network
        :param state_network_hidden_state_variables: number of dimensions in hidden representation of hidden network
        :param input_network_activation: nonlinear activation between linear operations in input network
                                         to leave-out nonlinear use `torch.nn.Identity`
        :param state_network_activation: nonlinear activation between linear operations in input network
        :param use_bias: if True bias vector is additional network learned parameter
                         applied according to LSTM formulas not to each intermediate linear layer
        """
        input_network = cls.create_operator_network(
            n_input_state_variables=n_input_state_variables,
            n_output_state_variables=n_output_state_variables,
            depth=input_network_depth,
            hidden_state_variables=input_network_hidden_state_variables,
            activation=input_network_activation,
        )

        state_network = cls.create_operator_network(
            n_input_state_variables=n_output_state_variables,  # input to state network needs to have hidden shape size
            n_output_state_variables=n_output_state_variables,
            depth=state_network_depth,
            hidden_state_variables=state_network_hidden_state_variables,
            activation=state_network_activation,
        )

        return cls(
            n_input_state_variables=n_input_state_variables,
            n_output_state_variables=n_output_state_variables,
            input_network=input_network,
            state_network=state_network,
            use_bias=use_bias,
        )

    @classmethod
    def create_operator_network(
        cls,
        n_input_state_variables: int,
        n_output_state_variables: int,
        depth: int,
        hidden_state_variables: int,
        activation: callable,
    ):
        layers = []
        layers.append(torch.nn.Linear(n_input_state_variables, hidden_state_variables))
        layers.append(activation)

        for _ in range(depth - 2):
            layers.append(torch.nn.Linear(hidden_state_variables, hidden_state_variables, bias=False))
            layers.append(activation)

        layers.append(torch.nn.Linear(hidden_state_variables, n_output_state_variables))
        return torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.input_network(inputs) + self.state_network(hidden) + self.bias


class GenericLSTM(torch.nn.Module):
    """
    Generic LSTM network with gates supporting any operation returning expected shapes
    Interface is given by `AbstractLSTMOperator` and typical LSTM operator is given by `ClassicLSTMOperator`
    """

    def __init__(
        self,
        n_input_state_variables: int,
        n_output_state_variables: int,
        operator: AbstractLSTMOperator,
        operator_kwargs: dict[str, Any] = None,
        return_states: bool = False,
        verbose_debug: bool = False,
        device: types.TorchDevice = "cpu",
    ):
        """
        :param n_input_state_variables: number of state input variables, equivalent to LSTM input size
        :param n_output_state_variables: number of states to produce, equivalent to LSTM hidden state size
        :param operator: operator applied to inputs and hidden state in each gate
        :param return_states: if True returns hidden state and context
        :param verbose_debug: if True verbose asserting of tensor shapes in performed (used to debug)
        :param device: torch device, can be `cpu` or `cuda`
        """
        super().__init__()
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.verbose_debug = verbose_debug
        self.return_states = return_states
        self.device = device

        self.input_operator = operator(n_input_state_variables, n_output_state_variables, **operator_kwargs).to(device)
        self.forget_operator = operator(n_input_state_variables, n_output_state_variables, **operator_kwargs).to(device)
        self.context_operator = operator(n_input_state_variables, n_output_state_variables, **operator_kwargs).to(
            device
        )
        self.output_operator = operator(n_input_state_variables, n_output_state_variables, **operator_kwargs).to(device)

    def verify_inputs(self, inputs):
        """
        Performs verbose verification of inputs z
            * dim must be equal to 3
            * features of input sequence must be equal to `n_input_state_variables`
        """
        dim_error_message = f"Number of input dimensions must be 3! {inputs.shape} != 3"
        input_shape_error_message = f"Invalid inputs! {inputs.shape[-1]} != {self.n_input_state_variables}"

        assert inputs.shape == 3, dim_error_message
        assert inputs.shape[-1] == self.n_input_state_variables, input_shape_error_message

    def verify_init_states(self, init_states: LSTMStatesType, batch_size: int) -> None:
        """
        Performs verbose verification of inputs
            * init_states must be two element tuple of tensors
            * hidden and context tensors must have have `(batch_size, n_output_state_variables)`
        """
        expected_shape = (batch_size, self.n_output_state_variables)

        shape_error_message = "Invalid {} shape! {} !=" + f"{expected_shape}"
        init_states_len_error_message = f"Init states must contain hidden and context tensors! {len(init_states)} != 2"

        assert len(init_states) == 2, init_states_len_error_message
        assert init_states[0].shape == expected_shape, shape_error_message.format("hidden", init_states[0].shape)
        assert init_states[1].shape == expected_shape, shape_error_message.format("context", init_states[1].shape)

    def set_initial_states(self, init_states: LSTMStatesType, batch_size: int) -> LSTMStatesType:
        """Sets initial states for hidden and context tensors"""
        if init_states is None:
            hidden = torch.zeros(batch_size, self.n_output_state_variables)
            context = torch.zeros(batch_size, self.n_output_state_variables)
        else:
            if self.verbose_debug:
                self.verify_init_states(init_states, batch_size)
            hidden, context = init_states
        return hidden.to(self.device), context.to(self.device)

    @staticmethod
    def process_tensor_sequence(sequence: list[torch.Tensor]) -> torch.Tensor:
        """Converts tensor list into single tensor. Used to process hidden_sequence to output tensor"""
        sequence = torch.cat(sequence, dim=0)
        return sequence.transpose(0, 1).contiguous()

    def forward(self, inputs: torch.Tensor, init_states: LSTMStatesType = None) -> LSTMOutputType:
        hidden_sequence = []

        if self.verbose_debug:
            self.verify_inputs(inputs)

        batch_size, n_time_steps, n_input_state_variables = inputs.size()
        hidden, context = self.set_initial_states(init_states, batch_size)

        for time_step_index in range(n_time_steps):
            inputs_at_time_step = inputs[:, time_step_index, :]

            input_gate_outputs = torch.sigmoid(self.input_operator(inputs_at_time_step, hidden))
            forget_gate_outputs = torch.sigmoid(self.forget_operator(inputs_at_time_step, hidden))
            context_gate_outputs = torch.tanh(self.context_operator(inputs_at_time_step, hidden))
            output = torch.sigmoid(self.output_operator(inputs_at_time_step, hidden))

            context = forget_gate_outputs * context + input_gate_outputs * context_gate_outputs
            hidden = output * torch.tanh(context)

            hidden_sequence.append(hidden.unsqueeze(dim=0))

        output_sequence = self.process_tensor_sequence(hidden_sequence)

        if self.return_states:
            return output_sequence, (hidden, context)

        return output_sequence
