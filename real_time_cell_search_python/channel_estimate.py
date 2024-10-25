import numpy as np
import queue
import copy


class ChannelEstimate(object):
    """
    ChannelEstimate actor class responsible for estimating the channel response
    based on Cell-specific Reference Signals (CRS) in LTE.
    """
    
    def __init__(self, name, actor_system, N_CP, N_CP_extra, Fs, N_dl_sc=12):
        """
        Initialize the ChannelEstimate actor with necessary parameters.
        
        Args:
            name (str): Name of the actor.
            actor_system (ActorSystem): Actor system managing this actor.
            N_CP (int): Cyclic prefix length.
            N_CP_extra (int): Extra CP length for the first OFDM symbol.
            Fs (float): Sampling frequency.
            N_dl_sc (int): Number of subcarriers per resource block (default 12).
        """
        self.name = name
        self.message_queue = queue.Queue()  # Queue to store incoming messages
        self.actor_system = actor_system

        self.ns = 0
        self.N_dl_sc = N_dl_sc
        self.N_CP = N_CP
        self.N_CP_extra = N_CP_extra
        self.Fs = Fs
        self.N_id = None 
        self.N_rb = None
        self.ant = None
        self.fd = None  
        
        self.k = []  # Subcarrier indices for CRS
        self.crs = []  # CRS values
        self.h_sym = []  # Channel estimates, antenna x available bw
        self.counter = 0
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
        Main processing method for handling messages related to channel estimation.
        """
        # Get the next message from the message queue
        message = self.message_queue.get()  # Get the next message
        print('Estimating Channels for {:}...'.format(message['destination']))

        self.update_sys(message)  # Update system state with the new message
        sym = self.strip_cp(message)  # Remove cyclic prefix
        sym = sym * np.exp(-2j * np.pi * message['fd'] * (np.arange(sym.size) + message['l'] * sym.size) / self.Fs)  # Frequency correction
        message['data'] = extract_OFDM(sym, self.N_rb)  # get the fourier transform of data and truncate available bandwidth

        # we only look at the case of number of antenna = 2
        if self.ant <= 2:
            if message['l'] in [0, 4]:
                message = self.estimate_h(message)  # Frequency domain channel estimation and interpolation
            else:
                message = self.intepolate(message)  # Time domain interpolation
        
        # Send message to the appropriate destination actor
        if message['destination'] == 'MIBDecode':
            self.actor_system.send_message('MIBDecode', copy.deepcopy(message)) # deep copy 'h'
            
        elif message['destination'] == 'SIB1Decode':
            self.actor_system.send_message('SIB1Decode', copy.deepcopy(message))
    
    def update_sys(self, message):
        """
        Update the system state based on incoming message parameters.
        """
        if self.N_rb != message['N_rb'] or self.ant != message['ant']:
            self.h_sym = np.zeros((message['ant'], message['N_rb'] * self.N_dl_sc), complex)

        if self.N_rb != message['N_rb']:
            self.N_rb = message['N_rb']

        if self.N_id != message['N_id']:
            self.N_id = message['N_id']

        if self.ant != message['ant']:
            self.ant = message['ant']
            self.k = [[]] * message['ant']
            self.crs = [[]] * message['ant']

        if message['l'] in [0, 4]:
            self.ns = message['ns']
            for p in range(self.ant):
                self.k[p], self.crs[p] = crs_in_re(p, message['l'], message['ns'], self.N_id, message['N_rb'])
    
    def strip_cp(self, message):
        """
        Remove the cyclic prefix from the OFDM symbol.
        """
        if message['l'] == 0:
            return message['data'][self.N_CP + self.N_CP_extra:]
        else:
            return message['data'][self.N_CP:]
    
    def estimate_h(self, message):
        """
        Estimate the channel response based on the CRS values.
        """
        l = message['l']
        re_sym = message['data']

        if l in [0, 4]:
            self.h_sym[0][self.k[0]] = re_sym[self.k[0]] * np.conj(self.crs[0])
            if self.ant == 2:
                self.h_sym[1][self.k[1]] = re_sym[self.k[1]] * np.conj(self.crs[1])

        # Interpolation in frequency domain
        for p in range(self.ant):
            for kk in range(self.N_rb * self.N_dl_sc):
                n = self.k[p][np.argmin(np.abs(self.k[p] - kk))]
                self.h_sym[p][kk] = self.h_sym[p][n]

        message['h'] = self.h_sym
        return message
    
    def intepolate(self, message):
        """
        Interpolate the channel estimates in the time domain.
        """
        message['h'] = self.h_sym
        return message





def extract_OFDM(ofdm_symbol, N_rb, N_rb_sc=12):
    """
    Extract active subcarriers from an OFDM symbol using FFT.

    Args:
        ofdm_symbol (numpy.ndarray): Time-domain OFDM symbol.
        N_rb (int): Number of resource blocks.
        N_rb_sc (int): Number of subcarriers per resource block (default 12).

    Returns:
        numpy.ndarray: Extracted active subcarriers from the frequency-domain OFDM symbol.
    """
    # Perform FFT and shift the result to center the active subcarriers
    re = np.fft.fftshift(np.fft.fft(ofdm_symbol))
    
    N_sc = N_rb * N_rb_sc  # Total number of active subcarriers
    N_FFT = ofdm_symbol.size  # FFT size

    # Indices of active subcarriers in the FFT result
    active_sc = np.concatenate((
        np.arange(N_FFT // 2 - N_sc // 2, N_FFT // 2),  # Left half of the active subcarriers
        np.arange(N_FFT // 2 + 1, N_FFT // 2 + N_sc // 2 + 1)  # Right half
    ))
    
    return re[active_sc]


def c_sequence(M, c_init, Nc=1600):
    """
    Generate a c-sequence (pseudo-random binary sequence) used in LTE.

    Args:
        M (int): Length of the sequence.
        c_init (int): Initialization value for the sequence.
        Nc (int): Offset for sequence generation (default 1600).

    Returns:
        numpy.ndarray: Generated c-sequence.
    """
    # Generate x_1 (m-sequence)
    x_1 = np.zeros(M + Nc, dtype=np.uint8)
    x_1[0] = 1
    for n in range(31, M + Nc):
        x_1[n] = x_1[n - 28] ^ x_1[n - 31]  # Generate based on previous values
    
    # Generate x_2 (initialized from c_init)
    x_2 = np.zeros(M + Nc, dtype=np.uint8)
    for n in range(31):
        x_2[n] = (c_init & (1 << n)) >> n
    for n in range(31, M + Nc):
        x_2[n] = x_2[n - 28] ^ x_2[n - 29] ^ x_2[n - 30] ^ x_2[n - 31]
    
    return x_1[Nc:] ^ x_2[Nc:]  # Return the sequence starting from Nc


# Generate Cell-specific Reference Signal (CRS) sequence for LTE
def crs_sequence(ns, l, N_id, N_cp=1, N_rb_max=110):
    """
    Generate the Cell-specific Reference Signal (CRS) sequence for LTE.

    Args:
        ns (int): Slot number within the radio frame.
        l (int): OFDM symbol index within the slot.
        N_id (int): Physical cell ID.
        N_cp (int): Cyclic prefix length (1 for normal CP, 0 for extended CP).
        N_rb_max (int): Maximum number of resource blocks (default 110).

    Returns:
        numpy.ndarray: CRS sequence for the given parameters.
    """
    # Compute c_init for the CRS sequence
    c_init = 2**10 * (7 * (ns + 1) + l + 1) * (2 * N_id + 1) + 2 * N_id + N_cp
    
    # Generate the c-sequence
    c = c_sequence(4 * N_rb_max, c_init)
    
    # Return CRS with sqrt(0.5) scaling
    return np.sqrt(0.5) * ((1 - 2.0 * c[0::2]) + 1j * (1 - 2.0 * c[1::2]))


def crs_in_re(p, l, ns, N_id, N_rb, N_rb_max=110):
    """
    Get CRS subcarrier indices and values for a specific antenna port and OFDM symbol.

    Args:
        p (int): Antenna port number.
        l (int): OFDM symbol index within the slot.
        ns (int): Slot number.
        N_id (int): Physical cell ID.
        N_rb (int): Number of resource blocks.
        N_rb_max (int): Maximum number of resource blocks (default 110).

    Returns:
        tuple: Subcarrier indices (k) and CRS values (crs).
    """
    # Define the subcarrier offset (nu) based on antenna port and symbol index
    nu = -1
    if p == 0 and l == 0:
        nu = 0
    elif p == 0 and l == 4:
        nu = 3
    elif p == 1 and l == 0:
        nu = 3
    elif p == 1 and l == 4:
        nu = 0
    elif p == 2 and l == 1:
        nu = 3 * (ns % 2)
    elif p == 3 and l == 1:
        nu = 3 + 3 * (ns % 2)

    if nu == -1:
        return (None, None)  # Invalid nu value, no CRS

    # Compute CRS frequency domain shift
    nu_shift = N_id % 6
    
    # Generate the CRS sequence
    cs = crs_sequence(ns, l, N_id)
    
    # Compute subcarrier indices and CRS values for the resource blocks
    m = np.arange(2 * N_rb)
    m_prime = m + N_rb_max - N_rb
    crs = cs[m_prime]  # CRS values
    
    k = 6 * m + (nu + nu_shift) % 6  # CRS subcarrier indices
    
    return (k, crs)


