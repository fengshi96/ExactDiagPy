import numpy as np


class pwavefunction(object):
    """ The wavefunction of a bipartite system, comes with trash elements.
    """

    def __init__(self, sys_dim, evn_dim, num_type='double'):
        """Creates an empty partioned wavefunction

        The wavefunction of a bipartite system, comes with trash elements.

        Parameters
        ----------
        sys_dim : The dimension of the Hilbert space of the system (int type)
        evn_dim : The dimension of the Hilbert space of the environment (int type)
        num_type : a double or complex
            The type of the wavefunction matrix elements.

        """
        super(pwavefunction, self).__init__()
        try:
            self.as_matrix = np.empty((sys_dim, evn_dim), num_type)
        except TypeError:
            print("Bad args for pwavefunction")
            raise

        self.left_dim = sys_dim
        self.right_dim = evn_dim
        self.num_type = num_type

    def rdm(self, block_to_be_traced_over):
        """
        Constructs the reduced density matrix of the wavefunction

        Parameters
        ----------
        block_to_be_traced_over : a string, left or right block to be traced over
        """

        if block_to_be_traced_over not in ('sys', 'evn'):
            print("block_to_be_traced_over must be left or right")
            raise

        if block_to_be_traced_over == 'sys':
            result = np.dot(np.transpose(self.as_matrix), self.as_matrix)
        else:
            result = np.dot(self.as_matrix, np.transpose(self.as_matrix))
        return result
