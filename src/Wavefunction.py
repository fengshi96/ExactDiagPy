import numpy as np


class Wavefunction(object):
    """ The wavefunction of a bipartite system, comes with trash elements.
    """

    def __init__(self, left_dim, right_dim, num_type='double'):
        """Creates an empty wavefunction

        The wavefunction of a bipartite system, comes with trash elements.

        Parameters
        ----------
        left_dim : The dimension of the Hilbert space of the left block (int type)
        right_dim : The dimension of the Hilbert space of the right block (int type)
        num_type : a double or complex
            The type of the wavefunction matrix elements.

        """
        super(Wavefunction, self).__init__()
        try:
            self.as_matrix = np.empty((left_dim, right_dim), num_type)
        except TypeError:
            print("Bad args for wavefunction")
            raise

        self.left_dim = left_dim
        self.right_dim = right_dim
        self.num_type = num_type

    def rdm(self, block_to_be_traced_over):
        """
        Constructs the reduced density matrix of the wavefunction

        Parameters
        ----------
        block_to_be_traced_over : a string, left or right block to be traced over
        """

        if block_to_be_traced_over not in ('left', 'right'):
            print("block_to_be_traced_over must be left or right")
            raise

        if block_to_be_traced_over == 'left':
            result = np.dot(np.transpose(self.as_matrix), self.as_matrix)
        else:
            result = np.dot(self.as_matrix, np.transpose(self.as_matrix))
        return result
