import numpy as np
import asn1tools
import queue
import channel_estimate as CE
# import matplotlib.pyplot as plt



class MIBDecode(object):
    """
    MIBDecode actor class responsible for decoding the Master Information Block (MIB)
    in LTE downlink using PBCH processing, Alamouti combining, QPSK demodulation,
    descrambling, rate matching, and convolutional decoding.
    """
    
    def __init__(self, name, actor_system):
        """
        Initialize the MIBDecode actor with necessary components.

        Args:
            name (str): Name of the actor.
            actor_system (ActorSystem): Actor system managing this actor.
        """
        self.name = name
        self.message_queue = queue.Queue()  # Queue to store incoming messages
        self.actor_system = actor_system
        
        self.fec = ConvCoder([0o133, 0o171, 0o165])  # Convolutional coder
        self.crc = CRC16_Table(0x1021)  # CRC table for CRC16
        asn1_file = '../lte-rrc-15.6.0.asn1'  # ASN.1 file path
        self.asn1 = asn1tools.compile_files(asn1_file, 'uper')  # Compile ASN.1 specification
        
        self.buffer = []  # Buffer to store symbols of PBCH for processing
        self.stop_processing = False  # Flag to stop processing

    def store_message(self, message):
        """
        Store a message in the actor's message queue.

        Args:
            message (dict): A message containing 'destination' and 'data'.
        """
        self.message_queue.put(message)

    def stop(self):
        """
        Stop processing further messages for this actor.
        """
        self.stop_processing = True 

    def __call__(self):
        """
        Main processing method for handling PBCH decoding messages.
        Combines symbols, demodulates, descrambles, applies rate matching,
        and decodes MIB information.
        """
        # Retrieve next message from the queue
        message = self.message_queue.get()  
        self.buffer.append(message)  # Append symbols to buffer for collecting PBCH

        if len(self.buffer) == 4:  # Process after collecting 4 symbols (Alamouti STC requires pairs)
            print('Decoding MIB...')

            # Combine received symbols with channel estimates for Alamouti decoding
            sym, h0, h1 = self.concatenate_and_extract_crs()
            sym = Alamouti_combine(sym, h0, h1)

            # # Plot the constellation
            # plt.plot(sym.real, sym.imag, '.')
            # plt.grid()
            # plt.axis('equal')
            # plt.xlabel('I')
            # plt.ylabel('Q')
            # plt.show()

            # Perform QPSK demodulation
            bits = qpsk_demodulate(sym)

            # Descramble PBCH
            ds_bits, rv = pbch_descrambling(1920, message['N_id'], bits)

            # Apply rate matching for PBCH
            coded_bits = pbch_rate_matching(ds_bits)

            # Decode using convolutional coding and Viterbi algorithm
            decoded_bits, cost = self.fec.decode(coded_bits, hamming_dist)

            # Check CRC and extract MIB
            mib_bytes, ant = pbch_crc_checking(decoded_bits, self.crc)
            mib_info = self.asn1.decode('MasterInformationBlock', mib_bytes)

            # Extract MIB parameters and send control information
            N_rb, sfn, c_phich = decode_mib(mib_info, rv)
            
            control_info = {
                'type': 'C',
                'destination': 'Controller',
                'source': 'MIBDecode',
                'N_rb': N_rb,
                'SFN': sfn,
                'C_phich': c_phich,
                'ant': ant
            }
            self.actor_system.send_message('Controller', control_info)

            # Reset the buffer after processing
            self.reset_buffer()

    def concatenate_and_extract_crs(self):
        """
        Concatenate symbols and extract CRS (Cell-specific Reference Signals) from buffer.

        Returns:
            tuple: Concatenated symbols (sym), h0 channel estimates, and h1 channel estimates.
        """
        sym, h0, h1 = [], [], []
        for message in self.buffer:
            sym = np.concatenate((sym, message['data']))
            h0 = np.concatenate((h0, message['h'][0]))
            h1 = np.concatenate((h1, message['h'][1]))

        # # Plot reshaped symbols for visual inspection
        # reshaped_sym = sym.reshape(4, 72)
        # plt.matshow(np.abs(reshaped_sym))
        # plt.xlabel('Subcarrier index (k)')
        # plt.ylabel('OFDM symbol index (l)')
        # plt.show()

        # plt.plot(np.abs(h0), label='|h0|')
        # plt.plot(np.abs(h1), label='|h1|')
        # plt.xlabel('Subcarrier index (k)')
        # plt.ylabel('|h|')
        # plt.legend()
        # plt.show()

        # Remove crs indices (e.g., index 0, 3, ...)
        ex_index = np.arange(0, 144, 3)
        sym = np.delete(sym, ex_index)
        h0 = np.delete(h0, ex_index)
        h1 = np.delete(h1, ex_index)
        
        return sym, h0, h1

    def reset_buffer(self):
        """Clear the buffer after processing a set of symbols."""
        self.buffer.clear()




def Alamouti_combine(r, H0, H1):
    """
    Combine two symbols using the Alamouti space-time code.

    Args:
        r (numpy.ndarray): Received symbol sequence.
        H0 (numpy.ndarray): Channel estimate for first transmit antenna.
        H1 (numpy.ndarray): Channel estimate for second transmit antenna.

    Returns:
        numpy.ndarray: Combined symbol sequence.
    """
    scale = abs(H0[0::2]) ** 2 + abs(H1[1::2]) ** 2
    d = np.zeros_like(r)

    # Alamouti decoding equations
    d[0::2] = (H0[0::2].conjugate() * r[0::2] + H1[1::2] * r[1::2].conjugate()) / scale
    d[1::2] = ((-H1[0::2].conjugate() * r[0::2] + H0[1::2] * r[1::2].conjugate()) / scale).conjugate()

    return d


def qpsk_demodulate(x):
    """
    Perform QPSK demodulation on a sequence of complex observations.

    Args:
        x (numpy.ndarray): Complex sequence to demodulate.

    Returns:
        numpy.ndarray: Demodulated bit sequence.
    """
    b = np.zeros(2 * len(x), dtype=np.uint8)

    for n in range(len(x)):
        b[2 * n] = 0 if np.real(x[n]) > 0 else 1
        b[2 * n + 1] = 0 if np.imag(x[n]) > 0 else 1

    return b


def pbch_descrambling(M_bit, N_id, bits):
    """
    Descramble the PBCH data using the cell ID.

    Args:
        M_bit (int): Length of the scrambled bit sequence.
        N_id (int): Physical cell ID.
        bits (numpy.ndarray): Bit sequence to descramble.

    Returns:
        tuple: Descrambled bits and redundancy version.
    """
    c = CE.c_sequence(M_bit, N_id)
    n_bits = len(bits)
    threshold = 0.9
    
    # Try each redundancy version (rv)
    for rv in range(4):
        ds_bits = bits ^ c[rv * n_bits:(rv + 1) * n_bits]
        if np.sum(ds_bits[:120] == ds_bits[120:240]) / 120 > threshold:
            break  # Select the redundancy version with the highest match

    return ds_bits, rv


def subblock_interleaver(seq, col_perm_table):
    """
    Apply subblock interleaving to a coded bit stream.

    Args:
        seq (numpy.ndarray): Sequence to interleave.
        col_perm_table (list): Column permutation table.

    Returns:
        numpy.ndarray: Interleaved bit sequence.
    """
    N_cc = 32
    D = len(seq)
    DUMMY = D + 10000

    # Calculate number of rows
    R = D // N_cc
    if R * N_cc < D:
        R += 1
    
    # Prepend dummy symbols
    N_dummy = R * N_cc - D
    y = np.concatenate((DUMMY * np.ones(N_dummy, dtype=seq.dtype), seq))

    # Reshape into a matrix for row-wise reading
    M = np.reshape(y, (R, N_cc))
    
    # Permute columns of the matrix
    P = np.zeros_like(M)
    for n in range(N_cc):
        P[:, n] = M[:, col_perm_table[n]]

    # Reshape and remove dummy symbols
    v = np.reshape(P.T, -1)
    return v[v != DUMMY]


def pbch_rate_matching(ds_bits):
    """
    Apply rate matching to PBCH decoded bits.

    Args:
        ds_bits (numpy.ndarray): Descrambled bits.

    Returns:
        numpy.ndarray: Rate-matched coded bits.
    """
    col_perm_table = np.array([1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
                               0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30])

    # Generate permutation table
    ind = np.arange(40)
    perm_table = subblock_interleaver(ind, col_perm_table)
    
    coded_bits = np.zeros((3, 40), dtype=np.uint8)
    for n in range(3):
        coded_bits[n, perm_table] = ds_bits[n * 40:(n + 1) * 40]
    
    return coded_bits


def count_ones(n):
    """
    Helper function to count the number of 1-bits in an integer.
    
    Args:
        n (int): Integer to count 1-bits from.
    
    Returns:
        int: Number of 1-bits.
    """
    b = 0
    while n != 0:
        b += n & 0x1  # Increment if LSB is 1
        n >>= 1  # Right shift to process next bit
    return b


def count_bits(n):
    """
    Helper function to count the total number of bits (1s and 0s) in an integer.
    
    Args:
        n (int): Integer to count total bits from.
    
    Returns:
        int: Total number of bits.
    """
    b = 0
    while n != 0:
        b += 0x1  # Increment for each bit
        n >>= 1  # Right shift to process next bit
    return b


def hamming_dist(obs, ref):
    """
    Compute the Hamming distance between two sequences.
    
    Args:
        obs (numpy.ndarray): Observed sequence.
        ref (numpy.ndarray): Reference sequence.
    
    Returns:
        int: Hamming distance between the two sequences.
    """
    return np.sum(ref != obs)  # Count mismatched bits


class ConvCoder(object):
    "Class to implement convolution FEC coding"
    
    def __init__(self, generators_list):
        """The convolutional code is described by a list of generator polynomials. Each 
        generator polynomial is summarized by an integer that captures the connections from
        the shift register to the coded bits.
        
        From the generators, the structural properties of the code can be deduced:
        * the number of bits in the generators determines the length of the shift register
        * the number of generators determines the code rate

        The standard specifies three specific generators (`0o133`, `0o171`, and `0o165`) 
        for a rate 1/3 code with a constraint length of 6 delays in the shift register. 
        This implementation is generic and supports arbitrary generators.
        """
        self.n_codes = len(generators_list)
        self.order = max([count_bits(n) for n in generators_list]) - 1
        self.generators = generators_list

        self.Nt = 2 ** (self.order + 1) # number of prossible outputs in each bit period
        self.Ns = 2 ** self.order       # number of states of the shift register

        # table of possible outputs
        self._t = np.empty((self.Nt, self.n_codes), dtype=np.uint8)
        for n in range(self.Nt):
            # loop over all possible bit patterns of length order+1
            for m in range(self.n_codes):
                self._t[n, m] = count_ones(n & self.generators[m]) & 0x1

    def decode(self, d, cost_fun):
        """Decode a sequence of observations using the Viterbi algorithm
        
        Inputs:
        * d - 2D array of observations; leading dimension must be equal to self.n_codes
        * cost_fun - function to measure similarity; use `Hamming_dist` for hard decision observations,
                     and `L2_dist` for soft decisions

        Returns:
        vector of bits; length is equal to the second dimension of `d`
        """
        assert d.shape[0] == self.n_codes, "Leading dimension of d must match number of generators"

        # initialization
        N = d.shape[1]             # number of input observations; also number of outputs
        costs = np.zeros(self.Ns)  # accumulated similarity metric
        # sequences of estimated bits; one for each register state
        survivors = 127 * np.zeros([self.Ns, N], dtype=np.uint8)

        for n in range(N):
            # temporary storage for the following loops; these are updated during the first phase
            # and then copied to the permanent variables in the second phase. This is needed
            # to prevent that intermediate results are overwritten prematurely
            tmp_cost = np.inf * np.ones(self.Ns)
            # tmp_ts = np.zeros(self.Ns, dtype=np.uint8)
            tmp_b = np.zeros(self.Ns, dtype=np.uint8)
            tmp_survivors = survivors[:, :n].copy()  # this copy is critical - grrrr
            
            obs = d[:, n]  # outputs for this bit period

            # update costs and survivor extensions; the key to the algorithm is
            # that there are possible 2^7 bit patterns in each bit period. The 6 most
            # significant bits form the state at the end of the bit period. The 6 least
            # significant bits define the state at the start of the period. Hence, there
            # are only two possible beginning states to reach an end state. For each end state
            # we only keep the path that with the smaller cost metric. 
            for te in np.arange(self.Ns, dtype=np.uint8):
                # loop over all states at the end of this bit period
                for b in np.arange(2, dtype=np.uint8):
                    # loop ver the LSB of the beginning states; this is the 
                    # only bit that's not also in te
                    # b is the LSB of t and ts
                    t = (te << 1) + b      # t combines the bits of ts and te 
                    ts = t & (self.Ns - 1) # state at the start of period
                    # compute cost recursively: cost at the start of the priod +
                    # cost associated with the difference betwee observation and
                    # coded bits for this transition from ts to te
                    c = costs[ts] + cost_fun(obs, self._t[t, :])
                    #if n == 1 and c == 0:
                    #    print(te, t, ts, c, survivors[ts, :n])
                    if c < tmp_cost[te]:
                        # store results if this is the lowest cost path to te
                        tmp_cost[te] = c
                        # capture the MSB of t (and te); that's the current bit
                        tmp_b[te] = (t & self.Ns) >> (self.order)
                        # tmp_ts[te] = ts
                        tmp_survivors[te, :] = survivors[ts, :n]

            # copy the updates to permanent variables for next iteration
            for te in np.arange(self.Ns, dtype=np.uint8):
                costs[te] = tmp_cost[te]
                survivors[te, :n] = tmp_survivors[te, :]
                survivors[te, n] = tmp_b[te]

                #if n == 1 and costs[te] == 0:
                #    print(te, tmp_ts[te], tmp_b[te], tmp_survivors[te, :], survivors[te, :n+1])
                
        # all done, find the lowest cost and return the corrponding survivor
        ind_survivor = np.argmin(costs)
        return survivors[ind_survivor, :], costs[ind_survivor]


    def encode(self, bits, init_state=None):
        """convolutional encoder
        
        Inputs:
        bits - information bits to be encoded
        init_state - initial state for the register (default: initialize via tail-biting; i.e., 
        use last elements of bits)

        Returns:
        2D - array of coded bits; leading dimension equals the number of generators, second
             dimension equals the number of bits
        """
        d = np.zeros((self.n_codes, len(bits)), dtype=np.uint8)

        # initialize state; default via tailbiting
        ts = np.uint8(0)
        if init_state is None:
            for n in range(self.order):
                ts += (bits[-(n+1)] << (self.order - n -1))
        else:
            ts[:] = init_state[:]
        
        # print("Initial state: {:06b}".format(ts))

        # Encoder
        for n in range(len(bits)):
            # construct transition t from state ts and next bit
            t = ts + (bits[n] << self.order)
            # look up output in table
            d[:,n] = self._t[t, :]
            # update state
            ts = (t >> 1)
            
            #print("n = {:d} - bit = {:d} state = {:06b} ({:d} {:d} {:d})".format(n, bits[n], ts, d[0,n], d[1,n], d[2,n]))

        return d     


class CRC16_Table(object):
    """
    Table-driven CRC16 calculation class based on a given polynomial.
    """
    
    def __init__(self, crc_poly):
        """
        Initialize the CRC16 lookup table for a given polynomial.
        
        Args:
            crc_poly (int): CRC polynomial.
        """
        self.crc_poly = crc_poly
        self._t = np.zeros(256, dtype=np.uint16)  # Initialize table
        mask = np.uint16(1 << 15)

        # Populate lookup table
        for n in np.arange(256, dtype=np.uint16):
            c = n << 8
            for k in range(8):
                if (c & mask) != 0:
                    c = crc_poly ^ (c << 1)
                else:
                    c <<= 1
            self._t[n] = c

    def _update_crc(self, byte, prev_crc):
        """
        Update the CRC for a single byte.

        Args:
            byte (int): Byte to process.
            prev_crc (int): Previous CRC value.

        Returns:
            int: Updated CRC value.
        """
        return self._t[((prev_crc >> 8) ^ byte) & 0xFF] ^ (prev_crc << 8) & 0xFFFF

    def __call__(self, data):
        """
        Compute the CRC for a given data array.

        Args:
            data (numpy.ndarray): Array of bytes (uint8) for which to compute the CRC.

        Returns:
            int: CRC value (0 indicates valid CRC).
        """
        crc = np.uint16(0)
        for d in data:
            crc = self._update_crc(d, crc)
        return crc & 0xFFFF


def pbch_crc_checking(decoded_bits, crc):
    """
    Perform CRC checking for PBCH decoded bits.

    Args:
        decoded_bits (numpy.ndarray): Decoded bit sequence.
        crc (function): CRC function to compute CRC16.

    Returns:
        tuple: Master Information Block (MIB) bytes and number of antennas (N_ant).
    """
    x_ant_d = {1: 0x00, 2: 0xFF, 4: 0x33}  # Antenna scrambling patterns
    bytes = np.packbits(decoded_bits)  # Convert bits to bytes
    crc_match = False

    # Try different antenna configurations for CRC checking
    for n_ant, x_ant in x_ant_d.items():
        for n in [3, 4]:
            bytes[n] ^= x_ant  # Apply scrambling
        if crc(bytes) == 0:  # Check if CRC is valid
            N_ant = n_ant
            mib_bytes = bytes[:3]  # Extract MIB bytes
            crc_match = True
            break

    assert crc_match, "CRC match failed."  # Ensure a CRC match was found
    return mib_bytes, N_ant



def decode_mib(mib_info, rv):
    """
    Decode the Master Information Block (MIB) information from decoded bits.

    Args:
        mib_info (dict): Decoded MIB information (ASN.1 decoded structure).
        rv (int): Redundancy version used in PBCH decoding.

    Returns:
        tuple: Contains the following:
            - N_rb (int): Number of resource blocks in the downlink.
            - sfn (int): System frame number.
            - c_phich (tuple): PHICH configuration (duration, resource).
    """
    # LTE bandwidth configuration table (mapping names to resource block counts)
    BW_TABLE = {'n6': 6, 'n15': 15, 'n25': 25, 'n50': 50, 'n75': 75, 'n100': 100}
    
    # Determine the number of resource blocks (N_rb) from the MIB information
    N_rb = BW_TABLE[mib_info['dl-Bandwidth']]
    
    # Extract system frame number (SFN) from the MIB information
    sfn_byte = mib_info['systemFrameNumber'][0]
    sfn_mib = int.from_bytes(sfn_byte, "big")
    sfn = sfn_mib * 4 + rv  # Adjust SFN with redundancy version
    
    # Extract PHICH configuration
    c_phich = (mib_info['phich-Config']['phich-Duration'], mib_info['phich-Config']['phich-Resource'])
    
    return N_rb, sfn, c_phich
