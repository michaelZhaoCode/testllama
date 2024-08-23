import torch


class linear_probe(torch.nn.Module):
    def __init__(self, input_dim, num_output_classes=2):
        super().__init__()

        if num_output_classes == 2:
            self.output_dim = 1
        else:
            self.output_dim = num_output_classes

        self.input_dim = input_dim

        self.linear_NN = torch.nn.Linear(
            self.input_dim, self.output_dim, bias=False
        )  # fully connected layer

        # self.Linear1 = torch.nn.Linear(
        #     self.input_dim, 100, bias=False
        # )
        # self.Linear2 = torch.nn.Linear(
        #     100, 50, bias=False
        # )
        # self.Linear3 = torch.nn.Linear(
        #     50, 25, bias=False
        # )
        # self.Linear4 = torch.nn.Linear(
        #     25, self.output_dim, bias=False
        # )

    def forward(self, x):
        """
        x: torch.tensor of shape (batch_size, LM_num_neurons)
        """

        linear_output = self.linear_NN(x)

        # relu = torch.nn.ReLU()
        # linear_output = x
        #
        # linear_output = relu(self.Linear1(linear_output))
        # linear_output = relu(self.Linear2(linear_output))
        # linear_output = relu(self.Linear3(linear_output))
        # linear_output = self.Linear4(linear_output)

        return linear_output
