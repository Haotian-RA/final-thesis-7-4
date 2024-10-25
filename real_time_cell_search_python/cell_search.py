import queue
import numpy as np




class CellSearch(object):
    """
    CellSearch actor class responsible for detecting and tracking the Primary and Secondary 
    Synchronization Signals (PSS/SSS) during cell search in LTE. It interacts with an actor system
    to send and receive messages related to frame synchronization and cell search results.
    """
    
    def __init__(self, name, actor_system, N_FFT, N_CP, Fs):
        """
        Initialize the CellSearch actor with the necessary parameters.

        Args:
            name (str): Name of the actor.
            actor_system (ActorSystem): The actor system managing this actor.
            N_FFT (int): FFT size used for OFDM symbols.
            N_CP (int): Length of the cyclic prefix.
            Fs (float): Sampling frequency.
        """
        self.name = name
        self.message_queue = queue.Queue()  # Queue to hold incoming messages
        self.actor_system = actor_system
        self.N_FFT = N_FFT
        self.N_CP = N_CP
        self.Fs = Fs  

        self.shift_reg = [None] * 2  # Shift register to store two consecutive OFDM symbols
        self.prior_sym = []  # Store the previous OFDM symbol
        self.nsym = 0  # Number of incoming symbols (messages) processed for first time cell search
        self.pss_start = None  # Start position of the PSS
        self.N_id_2 = None  # Cell ID part 2 from PSS detection
        
        self.threshold = 0.5  # Threshold for PSS detection
        self.pss, self.sss, self.pss_norm, self.sss_norm = pss_sss_norm(N_FFT)  # PSS and SSS signals and their norms
        
        self.track_frame = []  # Store frame for tracking PSS
        self.first_search = True

        self.stop_processing = False  # Flag to stop processing messages
        
    def store_message(self, message):
        """
        Store a message in the actor's message queue.

        Args:
            message (dict): A message containing 'destination' and 'data'. 'data' contains one ofdm symbol.
        """
        self.message_queue.put(message)

    def stop(self):
        """
        Stop processing further messages for this actor.
        """
        self.stop_processing = True   

    def __call__(self):
        """
        Main processing method of the actor. Handles messages and processes them
        based on the destination (either 'CellSearch' or 'FrameTrack').
        """
        # Get the next message from the message queue
        message = self.message_queue.get() 
        
        # If message is for CellSearch, perform PSS/SSS detection
        if message['destination'] == 'CellSearch': 
            if self.first_search:
                print('Initial Cell Searching...')
                
                self.nsym += 1 # number of processed ofdm symbol increased by 1
                self.prior_sym = self.shift_reg.pop(0)  # Move previous symbol out of the shift register
                self.shift_reg.append(message['data'])  # Append new symbol to the shift register
                
                # Perform PSS detection
                pss_nsym, pss_start, N_id_2 = self.match_pss()

                if pss_start:  # PSS detected successfully
                    fd = self.compute_fd()  # Compute frequency offset
                    N_id_1, F = self.match_sss()  # Perform SSS detection
                    
                    # Print detected PSS/SSS information
                    print(f'pss located at sample index: {(pss_nsym-1)*self.N_FFT + pss_start}, N_id_2: {N_id_2}')
                    print(f'Found the current slot in the {"first" if F == 0 else "second"} half frame, with N_id_1 = {N_id_1}')
                    
                    # Compute physical cell ID (PCI)
                    N_id = N_id_1 * 3 + N_id_2
                    
                    # Send control information to the Controller actor
                    control_info = {
                        'type': 'C',
                        'destination': 'Controller',
                        'source': 'CellSearch',
                        'pss_start': (pss_nsym-1)*self.N_FFT + pss_start,
                        'N_id': N_id,
                        'fd': fd
                    }
                    self.actor_system.send_message('Controller', control_info)
                    self.reset_pss()  # Reset the start of PSS position after detection
                    self.first_search = False

        # If message is for FrameTrack, perform frame tracking
        elif message['destination'] == 'FrameTrack':
            print('Periodically Tracking Frame...')
            
            self.track_frame = message['data']  # Store the frame for tracking
            _, pss_start, N_id_2 = self.match_pss(track=True)

            # threadhold may be too large.
            if not pss_start:
                # Reduce threshold if initial PSS match fails, but not below 0.3
                if self.threshold >= 0.3:
                    self.threshold -= 0.1
                    _, pss_start, N_id_2 = self.match_pss(track=True)

            if pss_start:  # PSS detected during tracking
                fd = self.compute_fd(track=True)
                N_id_1, _ = self.match_sss(track=True)
                
                # Print tracked PSS/SSS information
                print(f'pss is tracked at {pss_start} with a frequency offset {fd:.2f}')
                
                # Compute physical cell ID (PCI)
                N_id = N_id_1 * 3 + N_id_2
                print(f'N_id is: {N_id}')
                
                # Send control information to the Controller actor
                control_info = {
                    'type': 'C',
                    'destination': 'Controller',
                    'source': 'FrameTrack',
                    'pss_start': pss_start,
                    'N_id': N_id,
                    'fd': fd,
                    'start': message['start']
                }
                self.actor_system.send_message('Controller', control_info)
                self.reset_pss()  # Reset PSS state after tracking
                self.threshold = 0.5  # Reset detection threshold
    
    def match_pss(self, track=False):
        """
        Perform PSS (Primary Synchronization Signal) matching.

        Args:
            track (bool): Whether the function is called during frame tracking.

        Returns:
            tuple: Symbol number, PSS start index (in that symbol), N_id_2.
        """
        if not track:
            # Ensure both symbols in the shift register are valid before concatenation
            if any(sym is None for sym in self.shift_reg):
                return (self.nsym - 1, self.pss_start, self.N_id_2)
            sym_blocks = np.concatenate(self.shift_reg)
        else:
            sym_blocks = self.track_frame

        # PSS search process
        search = True
        ind = self.N_FFT
        max_corr = 0

        while search and ind < len(sym_blocks):
            sym = sym_blocks[ind - self.N_FFT:ind]  # Extract sliding symbol block

            # Check correlation with each PSS sequence (there are 3 possible PSS sequences)
            for n in range(3):
                corr = np.abs(np.sum(sym * np.conj(self.pss[n]))) / np.linalg.norm(sym) / self.pss_norm[n]
                if corr > self.threshold:
                    if corr > max_corr:
                        max_corr = corr
                    else:
                        search = False
                        self.N_id_2 = n
                        self.pss_start = ind - self.N_FFT - 1
            ind += 1

        return (self.nsym - 1, self.pss_start, self.N_id_2)
    
    def match_sss(self, track=False):
        """
        Perform SSS (Secondary Synchronization Signal) matching after PSS detection.

        Args:
            track (bool): Whether the function is called during frame tracking.

        Returns:
            tuple: N_id_1 and half-frame indicator F.
        """
        if not track:
            prior_blocks = np.concatenate((self.prior_sym, self.shift_reg[0]))
            sss_start = self.pss_start - self.N_CP
        else:
            prior_blocks = self.track_frame
            sss_start = self.pss_start - self.N_FFT - self.N_CP

        sss_sym = prior_blocks[sss_start:sss_start + self.N_FFT]
        corr = np.zeros(2 * 168)

        for m in range(2 * 168):
            corr[m] = np.abs(np.sum(sss_sym * np.conj(self.sss[self.N_id_2][m]))) / np.linalg.norm(sss_sym) / self.sss_norm[self.N_id_2][m]

        max_corr_ind = np.argmax(corr)
        self.N_id_1 = max_corr_ind // 2

        return (self.N_id_1, max_corr_ind % 2)
    
    def compute_fd(self, track=False):
        """
        Compute the frequency offset based on the matched PSS signal.

        Args:
            track (bool): Whether the function is called during frame tracking.

        Returns:
            float: Frequency offset.
        """
        sym_blocks = np.concatenate(self.shift_reg) if not track else self.track_frame
        pss_sym = sym_blocks[self.pss_start:self.pss_start + self.N_FFT]
        cross_corr = pss_sym * np.conj(self.pss[self.N_id_2])

        pl = np.sum(cross_corr[:self.N_FFT // 2])
        pu = np.sum(cross_corr[self.N_FFT // 2:])

        return np.angle(pu * np.conj(pl)) / (2 * np.pi * self.N_FFT // 2) * self.Fs
    
    def reset_pss(self):
        """
        Reset PSS start position after successful PSS detection.
        """
        self.pss_start = None
        
        
        
        
        
        
        
def seq_zadoff_chu(u):
    """
    Generate a Zadoff-Chu sequence for a given root index 'u'.
    
    Args:
        u (int): Root index used for the Zadoff-Chu sequence.
    
    Returns:
        numpy.ndarray: Zadoff-Chu sequence of length 63.
    """
    n = np.arange(63)  # Sequence index
    d_u = np.exp(-1j * np.pi * u * n * (n + 1) / 63)  # Zadoff-Chu sequence formula
    d_u[31] = 0  # Set the middle index to zero to satisfy LTE specification
    
    return d_u

def zadoff_chu(u, N_FFT):
    """
    Generate the time-domain Zadoff-Chu sequence for PSS using IFFT.
    
    Args:
        u (int): Root index for Zadoff-Chu sequence (PSS).
        N_FFT (int): FFT size.
    
    Returns:
        numpy.ndarray: Time-domain Zadoff-Chu sequence after IFFT.
    """
    zc = seq_zadoff_chu(u)  # Generate the Zadoff-Chu sequence
    
    # Create a zero-padded array with the Zadoff-Chu sequence centered in the frequency domain
    re = np.zeros(N_FFT, complex)
    re[N_FFT // 2 - 31: N_FFT // 2 + 32] = zc  # Place the sequence in the middle of FFT bins
    
    return np.fft.ifft(np.fft.ifftshift(re))  # Perform inverse FFT and return time-domain sequence


def tilde_s():
    """
    Generate the tilde_s sequence for SSS (based on m-sequence generation).
    
    Returns:
        numpy.ndarray: tilde_s sequence with values -1 and 1.
    """
    x = np.zeros(31, dtype=np.uint8)  # Initialize sequence
    x[4] = 1  # Set initial state
    for i in range(26):
        x[i + 5] = x[i + 2] ^ x[i]  # m-sequence generation using XOR
    
    return 1 - 2.0 * x  # Convert binary to bipolar values (-1, 1)

def tilde_c():
    """
    Generate the tilde_c sequence for SSS (based on m-sequence generation).
    
    Returns:
        numpy.ndarray: tilde_c sequence with values -1 and 1.
    """
    x = np.zeros(31, dtype=np.uint8)  # Initialize sequence
    x[4] = 1  # Set initial state
    for i in range(26):
        x[i + 5] = x[i + 3] ^ x[i]  # m-sequence generation using XOR
    
    return 1 - 2.0 * x  # Convert binary to bipolar values (-1, 1)

def tilde_z():
    """
    Generate the tilde_z sequence for SSS (based on m-sequence generation).
    
    Returns:
        numpy.ndarray: tilde_z sequence with values -1 and 1.
    """
    x = np.zeros(31, dtype=np.uint8)  # Initialize sequence
    x[4] = 1  # Set initial state
    for i in range(26):
        x[i + 5] = x[i + 4] ^ x[i + 2] ^ x[i + 1] ^ x[i]  # m-sequence generation using XOR
    
    return 1 - 2.0 * x  # Convert binary to bipolar values (-1, 1)


def m_01(N_id_1):
    """
    Compute m_0 and m_1 based on the physical layer cell ID N_id_1.

    Args:
        N_id_1 (int): Physical layer cell ID part 1.

    Returns:
        tuple: m_0 and m_1 values.
    """
    q_prime = N_id_1 // 30
    q = (N_id_1 + q_prime * (q_prime + 1) / 2) // 30
    m_prime = N_id_1 + q * (q + 1) / 2
    m_0 = int(m_prime % 31)  # Compute m_0 based on m'
    m_1 = int((m_0 + m_prime // 31 + 1) % 31)  # Compute m_1

    return m_0, m_1


def m_sequence(N_id_1, N_id_2, F, N_FFT):
    """
    Generate the complete SSS sequence based on the cell ID and frame index.

    Args:
        N_id_1 (int): Cell ID part 1.
        N_id_2 (int): Cell ID part 2.
        F (int): Half-frame indicator (0 or 1).
        N_FFT (int): FFT size.

    Returns:
        numpy.ndarray: Time-domain SSS sequence after IFFT.
    """
    # Compute m_0 and m_1 based on N_id_1
    m_0, m_1 = m_01(N_id_1)
    
    # Generate tilde sequences
    ts = tilde_s()
    tc = tilde_c()
    tz = tilde_z()
    
    # Generate c and s sequences shifted based on N_id_2 and m_0/m_1
    c_0 = np.roll(tc, -N_id_2)
    c_1 = np.roll(tc, -N_id_2 - 3)
    
    s_0 = np.roll(ts, -m_0)
    s_1 = np.roll(ts, -m_1)
    
    z_10 = np.roll(tz, -(m_0 % 8))
    z_11 = np.roll(tz, -(m_1 % 8))
    
    # Generate the SSS sequence based on half-frame F
    d = np.zeros(62)
    if F == 0:
        d[0::2] = s_0 * c_0
        d[1::2] = s_1 * c_1 * z_10
    elif F == 1:
        d[0::2] = s_1 * c_0
        d[1::2] = s_0 * c_1 * z_11
    
    # Map the SSS sequence to FFT bins
    re = np.zeros(N_FFT)
    re[N_FFT // 2 - 31:N_FFT // 2] = d[:31]
    re[N_FFT // 2 + 1:N_FFT // 2 + 32] = d[31:]
    
    return np.fft.ifft(np.fft.ifftshift(re))  # IFFT to generate time-domain signal


def pss_sss_norm(N_FFT):
    """
    Generate PSS and SSS sequences for all three possible root indexes and compute their norms.

    Args:
        N_FFT (int): FFT size.

    Returns:
        tuple: PSS sequences, SSS sequences, PSS norms, SSS norms.
    """
    pss = []  # Store PSS sequences
    pss_norm = []  # Store norms of PSS sequences
    sss = [[] for _ in range(3)]  # Store SSS sequences for each root index
    sss_norm = np.zeros((3, 2 * 168))  # Store norms of SSS sequences
    
    # Define the root indexes for PSS (u = 25, 29, 34 in LTE)
    u = [25, 29, 34]
    
    # Generate PSS and SSS sequences for each root index
    for n in range(3):
        # Generate PSS sequence for root index u[n]
        pss_t = zadoff_chu(u[n], N_FFT)
        pss.append(pss_t)
        pss_norm.append(np.linalg.norm(pss_t))  # Compute and store the norm
        
        # Generate SSS sequences for each combination of N_id_1 and frame indicator F
        for m in range(168):  # 168 possible N_id_1 values
            for F in range(2):  # Two possible half-frames
                sss[n].append(m_sequence(m, n, F, N_FFT))
                sss_norm[n, 2 * m + F] = np.linalg.norm(sss[n][m])  # Compute and store the norm

    return pss, sss, pss_norm, sss_norm
