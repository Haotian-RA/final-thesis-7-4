import queue
import asn1tools
import numpy as np
import mib_decoding as MD
import channel_estimate as CE
import matplotlib.pyplot as plt



class SIB1Decode(object):
    """
    Actor class for decoding System Information Block 1 (SIB1) in LTE using PBCH processing.
    Includes PCFICH, PDCCH, and PDSCH processing steps for extracting control and scheduling information.
    """
    
    def __init__(self, name, actor_system):
        """
        Initialize the SIB1Decode actor.

        Args:
            name (str): Name of the actor.
            actor_system (ActorSystem): Actor system managing this actor.
        """
        self.name = name
        self.message_queue = queue.Queue()
        self.actor_system = actor_system
        
        self.crc = MD.CRC16_Table(0x1021)
        self.fec = MD.ConvCoder([0o133, 0o171, 0o165])
        self.crc24 = CRC24A_Table(0x864CFB)  # CRC24 for PDSCH
        asn1_file = '../lte-rrc-15.6.0.asn1'
        self.asn1 = asn1tools.compile_files(asn1_file, 'uper')

        self.ns = 0  # LTE slot number (can modify for specific SIB1 decoding)
        self.N_rb = None
        self.N_id = None
        self.CFI = None  # Control Format Indicator
        self.TBS = None
        self.rv = None
        self.rnti = None
        self.pdsch_start = None
        self.pdsch_n_rb = None
        
        self.buffer = []  # Buffer to collect frames for decoding
        self.cfi_k = []  # PCFICH subcarrier indices

        self.stop_processing = False  # Flag to stop processing

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
        Main decoding function that handles incoming messages and processes SIB1.
        """
        # Get the next message from the message queue
        message = self.message_queue.get()
        self.buffer.append(message)  # Append message to buffer

        # Decode CFI from PCFICH
        if len(self.buffer) == 1:
            print('Decoding PCFICH...')

            self.update_sys(message)
            self.cfi_k = PCFICH_locs(self.N_rb, self.N_id)

            # Extract and combine PCFICH symbols
            pcfich_sym = message['data'][self.cfi_k]
            pcfich_h0 = message['h'][0][self.cfi_k]
            pcfich_h1 = message['h'][1][self.cfi_k]
            pcfich_sym = MD.Alamouti_combine(pcfich_sym, pcfich_h0, pcfich_h1)

            # QPSK demodulate and descramble
            pcfich_bits = MD.qpsk_demodulate(pcfich_sym)
            pcfich_ds_bits = pcfich_descrambling(self.ns, self.N_id, pcfich_bits)

            # Decode CFI value
            self.CFI = pcfich_decoding(pcfich_ds_bits, self.N_rb)
            print('CFI:', self.CFI)

        # Decode DCI from PDCCH if we have enough frames according to the CFI
        if len(self.buffer) == self.CFI:
            print('Decoding PDCCH...')
            
            # Collect PDCCH REs and combine
            data_blocks = [self.buffer[n] for n in range(len(self.buffer))]
            
            pdcch, h_pdcch = PDCCH_res(self.N_id, self.N_rb, self.CFI, data_blocks)
            pdcch_sym = MD.Alamouti_combine(np.array(pdcch), np.array(h_pdcch[0]), np.array(h_pdcch[1]))
            
            # # plot re
            # plt.plot(pdcch_sym.real,pdcch_sym.imag,'.')
            # plt.grid()
            # plt.axis('equal')
            # plt.xlabel('I')
            # plt.ylabel('Q')

            # plt.show()
            
            # QPSK demodulate, descramble, and rate match
            pdcch_bits = MD.qpsk_demodulate(pdcch_sym)
            pdcch_ds_bits = pdcch_descrambling(self.ns, self.N_id, pdcch_bits)
            pdcch_coded_bits = pdcch_rate_matching(pdcch_ds_bits)
            # Decode and check CRC for DCI
            dci_bits, cost = self.fec.decode(pdcch_coded_bits, MD.hamming_dist)
            dci, self.rnti = pdcch_crc_checking(dci_bits, self.crc)
            # Decode DCI to get PDSCH starting location and length
            self.pdsch_start, self.pdsch_n_rb = dci_decoding(dci, self.N_rb)
            self.TBS = 176
            self.rv = dci[-5] * 2 + dci[-4]
            print('DCI:',hex(int(''.join(map(str, dci)), 2)))

        # Decode PDSCH if we have the full subframe collected
        if len(self.buffer) == 14:
            print('Decoding SIB1...')
            pdsch, h0_pdsch, h1_pdsch = self.concatenate_and_extract_re(self.CFI, self.pdsch_start, self.pdsch_n_rb)
            pdsch_sym = MD.Alamouti_combine(pdsch, h0_pdsch, h1_pdsch)
            # QPSK demodulate, descramble, and rate match
            pdsch_bits = MD.qpsk_demodulate(pdsch_sym)
            pdsch_ds_bits = pdsch_descrambling(self.ns, self.N_id, pdsch_bits, self.rnti)
            pdsch_coded_bits = pdsch_rate_matching(pdsch_ds_bits, self.rv, self.TBS)
            # Convert coded bits to LLRs for Turbo decoding
            sib1_noise_sigma = 0.3
            pdsch_LLR = np.array([bits_to_LLRs(pdsch_coded_bits[n, :], sib1_noise_sigma) for n in range(3)])

            # Perform Turbo decoding
            f1, f2 = 13, 50
            sib1_LLRs = LTE_turbo_decoder(pdsch_LLR, self.TBS, f1, f2)

            # Make hard decision and pack bits to bytes
            pdsch_decoded_bytes = np.packbits(sib1_LLRs < 0)
            # Check CRC and decode ASN.1 structure
            sib1_bytes = pdsch_crc_checking(pdsch_decoded_bytes, self.crc24)
            sib1_info = self.asn1.decode('BCCH-DL-SCH-Message', sib1_bytes)

            # Send control information to the Controller actor
            control_info = {
                'type': 'C', 
                'destination': 'Controller', 
                'source': 'SIB1Decode', 
                'sib1_info': sib1_info
            }
            self.actor_system.send_message('Controller', control_info)
            self.reset_buffer()

    def update_sys(self, message):
        """
        Update system parameters based on the first message.
        """
        self.ns = message['ns']
        self.N_rb = message['N_rb']
        self.N_id = message['N_id']

    def concatenate_and_extract_re(self, CFI, pdsch_start, pdsch_n_rb, N_rb_sc=12):
        """
        Concatenate REs (Resource Elements) for PDSCH from the buffer.
        
        Args:
            CFI (int): Control Format Indicator.
            pdsch_start (int): Start position for PDSCH.
            pdsch_n_rb (int): Number of resource blocks for PDSCH.
            N_rb_sc (int): Subcarriers per resource block (default is 12).
        
        Returns:
            tuple: Concatenated symbols and channel estimates (sym, h0, h1).
        """
        sym, h0, h1 = [], [], []
        for l in range(CFI, 14):
            sym = np.concatenate((sym, self.buffer[l]['data'][pdsch_start * N_rb_sc:(pdsch_start + pdsch_n_rb) * N_rb_sc]))
            h0 = np.concatenate((h0, self.buffer[l]['h'][0][pdsch_start * N_rb_sc:(pdsch_start + pdsch_n_rb) * N_rb_sc]))
            h1 = np.concatenate((h1, self.buffer[l]['h'][1][pdsch_start * N_rb_sc:(pdsch_start + pdsch_n_rb) * N_rb_sc]))

        # Remove CRS (Cell-specific Reference Signals) for symbols 4, 7, and 11
        ex_index4 = np.arange((4 - CFI) * N_rb_sc * pdsch_n_rb, (5 - CFI) * N_rb_sc * pdsch_n_rb, 3)
        ex_index7 = np.arange((7 - CFI) * N_rb_sc * pdsch_n_rb, (8 - CFI) * N_rb_sc * pdsch_n_rb, 3)
        ex_index11 = np.arange((11 - CFI) * N_rb_sc * pdsch_n_rb, (12 - CFI) * N_rb_sc * pdsch_n_rb, 3)
        ex_index = np.concatenate((ex_index4, ex_index7, ex_index11))

        sym = np.delete(sym, ex_index)
        h0 = np.delete(h0, ex_index)
        h1 = np.delete(h1, ex_index)

        return sym, h0, h1

    def reset_buffer(self):
        """
        Clear the buffer after processing a full subframe.
        """
        self.buffer.clear()
        



def PCFICH_locs(N_RB, N_id):
    """
    Compute subcarrier indices for PCFICH (Physical Control Format Indicator Channel).

    Args:
        N_RB (int): Number of resource blocks.
        N_id (int): Cell ID.

    Returns:
        numpy.ndarray: Subcarrier indices for PCFICH.
    """
    k_ind = np.zeros(16, dtype=int)
    N_SC_RB = 12  # Subcarriers per resource block
    N_SC_RB_2 = N_SC_RB // 2

    k_bar = N_SC_RB_2 * (N_id % (2 * N_RB))  # Compute base index

    m = 0
    for n in range(4):  # Loop over 4 quadruplets
        k_init = k_bar + ((n * N_RB) // 2) * N_SC_RB_2

        for i in range(6):
            if (i % 3) != 0:  # Skip CRS (Cell-specific Reference Signal) subcarriers
                k_ind[m] = (k_init + i) % (N_RB * N_SC_RB)
                m += 1

    return k_ind


def pcfich_descrambling(ns, N_id, bits):
    """
    Descramble PCFICH bits using a sequence generated with `c_sequence`.

    Args:
        ns (int): Slot number.
        N_id (int): Cell ID.
        bits (numpy.ndarray): Bits to descramble.

    Returns:
        numpy.ndarray: Descrambled bits.
    """
    c_init = (((ns // 2 + 1) * (2 * N_id + 1)) << 9) + N_id  # Initial value for scrambling sequence
    c = CE.c_sequence(len(bits), c_init)  # Generate scrambling sequence
    return bits ^ c  # Return descrambled bits


def pcfich_decoding(bits, N_rb_dl):
    """
    Decode PCFICH (Physical Control Format Indicator Channel) to obtain the Control Format Indicator (CFI).

    Args:
        bits (numpy.ndarray): Demodulated and descrambled PCFICH bits.
        N_rb_dl (int): Number of downlink resource blocks.

    Returns:
        int: Control Format Indicator (CFI) value.
    """
    CFI_code_table = {
        1: np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.uint8),
        2: np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8),
        3: np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.uint8),
    }

    dist = np.zeros(3)
    CFI_vec = np.zeros(3, dtype=int)

    # Calculate Hamming distances for each possible CFI pattern
    for n, (cfi_val, cfi_seq) in enumerate(CFI_code_table.items()):
        dist[n] = MD.hamming_dist(bits, cfi_seq)
        CFI_vec[n] = cfi_val

    # Choose the CFI with minimum Hamming distance
    obs_CFI = CFI_vec[np.argmin(dist)]
    CFI = obs_CFI + 1 if N_rb_dl <= 10 else obs_CFI  # Adjust CFI for low bandwidth

    return CFI

def PDCCH_res(N_id,N_rb_dl,CFI,data_blocks):
    """
    Extract resource elements (RE) for PDCCH, accounting for exclusions like PHICH and PCFICH.

    Args:
        N_id (int): Cell ID.
        N_rb_dl (int): Number of downlink resource blocks.
        CFI (int): Control Format Indicator.
        data_blocks (list): List of data blocks containing PDCCH symbols and channel estimates.

    Returns:
        tuple: PDCCH symbols and corresponding channel estimates.
    """
    # exclude PHICH in sym0
    pdcch_available_reg_l1 = {n for n in range(2*N_rb_dl)}

    # PCFICH REGs are fairly easy 
    PCFI_REG = {(N_id % (2*N_rb_dl) + ((n*N_rb_dl) // 2) % (2*N_rb_dl)) for n in range(4)}

    # this is what's left for PHICH and PDCCH
    pdcch_available_reg_l1.difference_update(PCFI_REG)
    pdcch_available_reg_l1_vec = np.sort(np.array([n for n in pdcch_available_reg_l1]))
    n_0 = len(pdcch_available_reg_l1)
    # now deal with the PHICH
    PHICH_REG = set()

    # note: this is specific to FDD and normal CP
    Ng = 1   # this reflects phich-Resource from MIB
    N_group = (Ng * N_rb_dl) // 8
    if 8*N_group < Ng * N_rb_dl:
        N_group += 1

    for m in range(N_group):
        for i in range(3):
            n_i = (N_id + m + ((i*n_0) // 3)) % n_0 # index into pdcch_available_reg_l1_vec
            reg = pdcch_available_reg_l1_vec[n_i]   # the actual REG
            
            # move REG from pdcch_available_reg_l1 to PHICH_REG
            pdcch_available_reg_l1.discard(reg)
            PHICH_REG.add(reg)
    
    # Extract REGs of PDCCH
    pdcch_reg_l1 = [(0,6*i) for i in list(pdcch_available_reg_l1)]
    pdcch_reg_l2 = [(1,4*i) for i in np.arange(N_rb_dl*3)] # for two antennas
    pdcch_reg_l3 = [(2,4*i) for i in np.arange(N_rb_dl*3)]
    pdcch_reg = []

    if CFI == 1:
        pdcch_reg = pdcch_reg_l1
    elif CFI >= 2:
        if CFI == 2:
            all_reg = pdcch_reg_l1 + pdcch_reg_l2   
        elif CFI == 3:
            all_reg = pdcch_reg_l1 + pdcch_reg_l2 + pdcch_reg_l3    

        for k in range(N_rb_dl*12):
            l = 0
            while l < CFI:
                if (l,k) in all_reg:
                    if l == 0:
                        pdcch_reg.append((l,k))
                    elif l > 0:
                        pdcch_reg.append((l,k))
                l += 1
                
    pdcch_reg = np.array(pdcch_reg)
    N_reg = len(pdcch_reg)

    # de-cyclic shift
    pdcch_reg_cs = np.roll(pdcch_reg,(N_id%N_reg),axis=0)
    
    col_perm_table = np.array([1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 
                            0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30])

    # de-interleaving
    ind  = np.arange(N_reg)                   
    perm_table = MD.subblock_interleaver(ind,col_perm_table)

    pdcch_num_reg = np.zeros_like(pdcch_reg)
    pdcch_num_reg[perm_table] = pdcch_reg_cs
    
    # extract re of PDCCH
    pdcch = []
    h_pdcch = [[],[]]

    for (l,k) in pdcch_num_reg:
        if l == 0:
            ind = [1,2,4,5]
        else:
            ind = np.arange(4)
        
        for m in ind:
            pdcch.append(data_blocks[l]['data'][k+m])
            for p in range(2):
                h_pdcch[p].append(data_blocks[l]['h'][p][k+m])
                
    return (pdcch,h_pdcch)  


def pdcch_descrambling(ns, N_id, bits):
    """
    Descramble PDCCH bits using the LTE-defined scrambling sequence.

    Args:
        ns (int): Slot number.
        N_id (int): Cell ID.
        bits (numpy.ndarray): Bits to descramble.

    Returns:
        numpy.ndarray: Descrambled bits.
    """
    c_init = ((ns // 2) << 9) + N_id
    c = CE.c_sequence(len(bits), c_init)
    return bits ^ c


def pdcch_rate_matching(bits):
    """
    Perform rate matching for PDCCH bits based on observed repetition length.

    Args:
        bits (numpy.ndarray): Received PDCCH bits.

    Returns:
        numpy.ndarray: Coded PDCCH bits with rate matching.
    """
    wl = 40
    bits_f40 = bits[:wl]
    for n in range(bits.size):
        if np.all(bits_f40 == bits[n + 1:n + 1 + 40]):
            l_dci_3x = n + 1
            break
    l_dci = l_dci_3x // 3

    # Permutation table and subblock interleaving
    col_perm_table = np.array([1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 
                               0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30])
    perm_table = MD.subblock_interleaver(np.arange(l_dci), col_perm_table)
    coded_bits = np.zeros((3, l_dci), dtype=np.uint8)
    for n in range(3):
        coded_bits[n, perm_table] = bits[n * l_dci:(n + 1) * l_dci]

    return coded_bits


def pdcch_crc_checking(dci_bits_with_crc, crc):
    """
    Check CRC for PDCCH and find the corresponding RNTI.

    Args:
        dci_bits_with_crc (numpy.ndarray): DCI bits with CRC.
        crc (function): CRC check function.

    Returns:
        tuple: DCI bits and RNTI if CRC matches.
    """
    n_dci = len(dci_bits_with_crc) - 16
    dci_bits = dci_bits_with_crc[:n_dci]
    crc_match = False

    for rnti in range(0xFFFF + 1):
        parity_bits = dci_bits_with_crc[n_dci:] ^ [int(b) for b in bin(rnti)[2:].zfill(16)]
        pdcch_crc_bits = np.concatenate([dci_bits, parity_bits])
        if crc(np.packbits(pdcch_crc_bits)) == 0:
            crc_match = True
            break

    assert crc_match, "CRC check failed."
    return dci_bits, rnti


def dci_decoding(dci, N_rb):
    """
    Decode DCI bits to extract the start position and number of resource blocks.

    Args:
        dci (numpy.ndarray): DCI bits.
        N_rb (int): Number of resource blocks.

    Returns:
        tuple: Start position and length of PDSCH.
    """
    riv_bin = dci[2:13]
    riv = riv_bin.dot(2 ** np.arange(riv_bin.size)[::-1])
    return riv % N_rb, riv // N_rb + 1


def pdsch_descrambling(ns, N_id, bits, rnti):
    """
    Descramble PDSCH bits using a scrambling sequence based on ns, N_id, and rnti.

    Args:
        ns (int): Slot number.
        N_id (int): Cell ID.
        bits (numpy.ndarray): Encoded bits to descramble.
        rnti (int): Radio Network Temporary Identifier.

    Returns:
        numpy.ndarray: Descrambled bits.
    """
    q = 0  # Number of codewords; for PDSCH, typically 0 or 1.
    c_init = (rnti << 14) + (q << 13) + ((ns // 2) << 9) + N_id  # Initialize sequence
    c = CE.c_sequence(len(bits), c_init)  # Generate scrambling sequence
    return bits * (1.0 - 2 * c)  # Apply scrambling sequence for descrambling


def pdsch_rate_matching(ds_bits, rv, TBS):
    """
    Rate matching for PDSCH bits to handle retransmissions and error resilience.

    Args:
        ds_bits (numpy.ndarray): Input bits after descrambling.
        rv (int): Redundancy version.
        TBS (int): Transport block size.

    Returns:
        numpy.ndarray: Rate-matched coded bits.
    """
    # Initialize permutation table and parameters for interleaving
    col_perm_table = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                               1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])
    CRC_len = 24
    trellis_end = 4
    coded_block_size = TBS + CRC_len + trellis_end

    # Calculate interleaving parameters
    seq = np.arange(coded_block_size)
    C = 32
    D = len(seq)
    DUMMY = D + 10000
    R = (D + C - 1) // C  # Calculate required rows

    # Fill and interleave columns
    K_pi = R * C
    N_dummy = K_pi - D
    y = np.concatenate((DUMMY * np.ones(N_dummy, dtype=seq.dtype), seq))
    M = np.reshape(y, (R, C))
    
    # Permute columns according to col_perm_table
    p = np.zeros_like(M)
    for n in range(C):
        p[:, n] = M[:, col_perm_table[n]]
    v = np.reshape(p.T, -1)

    # Handle third port permutation and create combined vector
    k = np.arange(K_pi)
    pi_k = (col_perm_table[k // R] + C * (k % R) + 1) % K_pi
    v3 = y[pi_k]

    # Combine blocks with dummy symbols filtered
    v1, v2 = v.copy(), v.copy()
    v2[v2 < DUMMY] += D
    v3[v3 < DUMMY] += 2 * D
    v_comb = np.zeros(3 * K_pi, int)
    v_comb[:K_pi] = v1
    v_comb[K_pi::2] = v2
    v_comb[K_pi + 1::2] = v3

    # Select elements based on redundancy version
    k0 = int(R * (2 * np.ceil(len(v_comb) / (8 * R)) * rv + 2))
    e = np.zeros(3 * D, int)
    k, j = 0, 0
    while k < 3 * D:
        if v_comb[(k0 + j) % len(v_comb)] != DUMMY:
            e[k] = v_comb[(k0 + j) % len(v_comb)]
            k += 1
        j += 1

    # Build final rate-matched bits
    collect_len = 3 * coded_block_size
    pdsch_coded_bits = np.zeros(collect_len)
    pdsch_coded_bits[e] = ds_bits[:collect_len]
    return pdsch_coded_bits.reshape(3, coded_block_size)


def bits_to_LLRs(bits, noise_sigma):
    """
    Convert bits to Log-Likelihood Ratios (LLRs) for QPSK modulation in an AWGN channel.

    Args:
        bits (numpy.ndarray): Received bits.
        noise_sigma (float): Standard deviation of noise.

    Returns:
        numpy.ndarray: LLRs of bits.
    """
    return -2 * bits / noise_sigma**2


def turbo_len(transport_size):
    """
    Calculate the length of Turbo-encoded data given a transport size.

    Args:
        transport_size (int): Transport block size.

    Returns:
        int: Total length after Turbo encoding.
    """
    pdsch_crc_len = 24  # PDSCH CRC length in LTE
    turbo_tail = 4  # Additional bits for Turbo coding
    return transport_size + pdsch_crc_len + turbo_tail


def turbo_permutation(K, f1, f2):
    """
    Perform turbo code permutation.

    Args:
        K (int): Length of sequence to permute.
        f1 (int): First turbo permutation parameter.
        f2 (int): Second turbo permutation parameter.

    Returns:
        numpy.ndarray: Permutation indices.
    """
    i = np.arange(K)
    return (f1 * i + f2 * i**2) % K


def turbo_permutation_inv(K, f1, f2):
    """
    Inverse of the turbo code permutation.

    Args:
        K (int): Length of sequence to permute.
        f1 (int): First turbo permutation parameter.
        f2 (int): Second turbo permutation parameter.

    Returns:
        numpy.ndarray: Indices for the inverse permutation.
    """
    fwd = list(turbo_permutation(K, f1, f2))
    return np.array([fwd.index(i) for i in range(K)])


def maxstar(a, b):
    """
    Max-star operation to compute the max* between two log-domain values.

    Args:
        a (numpy.ndarray): Array of log values.
        b (numpy.ndarray): Array of log values.

    Returns:
        numpy.ndarray: Result of the max-star operation.
    """
    result = np.empty_like(a)
    ainf, binf = a == -np.inf, b == -np.inf
    result[ainf] = b[ainf]
    result[binf] = a[binf]
    noninf = ~ainf & ~binf
    an, bn = a[noninf], b[noninf]
    result[noninf] = np.maximum(an, bn) + np.log1p(np.exp(-np.abs(an - bn)))
    return result

def BCJR(Ru, R, Ap):
    """
    BCJR Algorithm for Turbo Decoding in LTE.

    Args:
        Ru (numpy.ndarray): Log-likelihood ratios (LLRs) for code 1.
        R (numpy.ndarray): Log-likelihood ratios (LLRs) for code 2.
        Ap (numpy.ndarray): A priori information LLRs.

    Returns:
        numpy.ndarray: Extrinsic information LLRs for decoded bits.
    """
    T = Ru.size # time steps
    nu = 3 # number of states = 2^nu
    Gamma = np.empty((T, 2**nu, 2**nu))
    Gamma[...] = -np.inf
    A = np.empty((T, 2**nu))
    A[...] = -np.inf
    A[0, 0] = 0
    B = np.empty((T, 2**nu))
    B[...] = -np.inf
    B[-1, 0] = 0
    
    # States are encoded as the concatenation of the
    # bits in the shift register, with the oldest bit
    # in the LSB, so that the shift register is shifted
    # left in each epoch.
    for r in range(2**nu):
        for s in range(2**nu):
            # non-termination trellis
            if r >> 1 == s & (2**(nu-1)-1):
                # S_r to S_s is a trellis edge
                Gamma[:-nu, r, s] = 0
                feedback = (r ^ (r >> 1)) & 1
                # The information bit corresponding to the
                # transition from S_r to S_s
                newbit = (s >> (nu - 1)) ^ feedback
                if newbit == 0:
                    Gamma[:-nu, r, s] += Ap - np.log1p(np.exp(Ap))
                else:
                    Gamma[:-nu, r, s] += -Ap - np.log1p(np.exp(-Ap))
                c0 = newbit
                c1 = (r ^ (r >> 2) ^ (s >> (nu - 1))) & 1                
                if c0 == 0:
                    Gamma[:-nu, r, s] += Ru[:-nu] - np.log1p(np.exp(Ru[:-nu]))
                else:
                    Gamma[:-nu, r, s] += -Ru[:-nu] - np.log1p(np.exp(-Ru[:-nu]))
                if c1 == 0:
                    Gamma[:-nu, r, s] += R[:-nu] - np.log1p(np.exp(R[:-nu]))
                else:
                    Gamma[:-nu, r, s] += -R[:-nu] - np.log1p(np.exp(-R[:-nu]))
            # termination trellis
            if r >> 1 == s:
                # S_r to S_s is a trellis edge
                Gamma[-nu:, r, s] = 0
                feedback = (r ^ (r >> 1)) & 1
                c0 = feedback
                c1 = (r ^ (r >> 2)) & 1
                if c0 == 0:
                    Gamma[-nu:, r, s] += Ru[-nu:] - np.log1p(np.exp(Ru[-nu:]))
                else:
                    Gamma[-nu:, r, s] += -Ru[-nu:] - np.log1p(np.exp(-Ru[-nu:]))
                if c1 == 0:
                    Gamma[-nu:, r, s] += R[-nu:] - np.log1p(np.exp(R[-nu:]))
                else:
                    Gamma[-nu:, r, s] += -R[-nu:] - np.log1p(np.exp(-R[-nu:]))

    # Note: For A, t is numbered from 0 to T-1,
    # while for Gamma and B, it is numbered from 1 to T
    for t in range(1, T):
        for r in range(2**nu):
            A[t, :] = maxstar(A[t, :], A[t-1, r] + Gamma[t-1, r, :])

    for t in range(T-2, -1, -1):
        for s in range(2**nu):
            B[t, :] = maxstar(B[t, :], B[t+1, s] + Gamma[t+1, :, s])
    
    M = A[:, :, np.newaxis] + Gamma + B[:, np.newaxis, :]

    Lp = np.empty_like(Ap)
    Lm = np.empty_like(Ap)
    Lp[:] = -np.inf
    Lm[:] = -np.inf
    for r in range(2**nu):
        for s in range(2**nu):
            # always non-termination trellis
            if r >> 1 == s & (2**(nu-1)-1):
                feedback = (r ^ (r >> 1)) & 1
                newbit = (s >> (nu - 1)) ^ feedback
                if newbit == 1:
                    Lp = maxstar(Lp, M[:-nu, r, s])
                else:
                    Lm = maxstar(Lm, M[:-nu, r, s])
    return Lm - Lp

def turbo_decoder(Ru1, R1, Ru2, R2, f1, f2, num_iterations=50, maxA=512, maxR=512, do_plots=False):
    """
    Turbo decoder function using the BCJR algorithm.

    Args:
        Ru1, R1, Ru2, R2 (numpy.ndarray): LLR arrays for each turbo code input.
        f1, f2 (int): Permutation parameters.
        num_iterations (int): Maximum number of iterations for decoding.
        maxA, maxR (float): Limits on the LLR values.
        do_plots (bool): Flag for plotting intermediate results.

    Returns:
        numpy.ndarray: Final LLRs after decoding.
    """
    for R in [Ru1, R1, Ru2, R2]:
        R[:] = np.clip(R, -maxR, maxR)
    K = Ru1.size - 3
    E2 = np.zeros(K)
    for iteration in range(num_iterations):
        A1 = E2[turbo_permutation_inv(K,f1,f2)]
        A1 = np.clip(A1, -maxA, maxA)
        L1 = BCJR(Ru1, R1, A1)
        E1 = L1 - Ru1[:-3] - A1        
        A2 = E1[turbo_permutation(K,f1,f2)]
        A2 = np.clip(A2, -maxA, maxA)
        L2 = BCJR(Ru2, R2, A2)
        E2 = L2 - Ru2[:-3] - A2
        if do_plots:
            fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
            axs[0].plot(L1, '.')
            axs[1].plot(L2, '.', color='C1')
            axs[0].set_ylabel('L1')
            axs[1].set_ylabel('L2')
            fig.suptitle(f'LTE Turbo decoder iteration {iteration}', y=0.93)
            axs[1].set_xlabel('Message bit')
            plt.subplots_adjust(hspace=0)
    return L2[turbo_permutation_inv(K,f1,f2)]

def LTE_turbo_decoder(LLRs, transport_size, f1, f2, plot_channel_LLRs=False, **kwargs):
    """
    LTE Turbo decoder with standard LTE interleaver structure.

    Args:
        LLRs (numpy.ndarray): Channel LLRs.
        transport_size (int): Size of the transport block.
        f1, f2 (int): Turbo interleaver parameters.
        plot_channel_LLRs (bool): Option to plot channel LLRs.

    Returns:
        numpy.ndarray: Final decoded bits.
    """
    if plot_channel_LLRs:
        plt.figure()
        plt.plot(LLRs.ravel(), '.')
        plt.title('LTE Turbo decoder channel LLRs')
    # Section 5.1.3.2.2 in TS 36.212 for trellis termination "shuffling"
    K = turbo_len(transport_size) - 4
    # x
    Ru1 = np.concatenate([LLRs[0, :K], [LLRs[0, K], LLRs[2, K], LLRs[1, K+1]]])
    # z
    R1 = np.concatenate([LLRs[1, :K], [LLRs[1, K], LLRs[0, K+1], LLRs[2, K+1]]])
    # x'
    Ru2 = np.concatenate([LLRs[0, :K][turbo_permutation(K,f1,f2)], [LLRs[0, K+2], LLRs[2, K+2], LLRs[1, K+3]]])
    # z'
    R2 = np.concatenate([LLRs[2, :K], [LLRs[1, K+2], LLRs[0, K+3], LLRs[2, K+3]]])
    return turbo_decoder(Ru1, R1, Ru2, R2, f1, f2, **kwargs)

class CRC24A_Table:
    """
    Table-driven CRC computation for LTE's CRC-24A polynomial.
    """

    def __init__(self, crc_poly):
        self.crc_poly = crc_poly
        self._t = np.zeros(256, dtype=np.uint32)
        mask = np.uint32(1 << 23)
        
        for n in range(256):
            c = np.uint32(n << 16)
            for _ in range(8):
                c = self.crc_poly ^ (c << 1) if c & mask else c << 1
            self._t[n] = c & 0xFFFFFF

    def _update_crc(self, byte, prev_crc):
        index = ((prev_crc >> 16) ^ byte) & 0xFF
        return ((prev_crc << 8) ^ self._t[index]) & 0xFFFFFF

    def __call__(self, data):
        crc = np.uint32(0)
        for byte in data:
            crc = self._update_crc(byte, crc)
        return crc


def pdsch_crc_checking(decoded_bytes, crc):
    """
    CRC checking for PDSCH payload.

    Args:
        decoded_bytes (numpy.ndarray): Decoded bytes from PDSCH.
        crc (CRC24A_Table): CRC instance.

    Returns:
        numpy.ndarray: Payload without CRC if it passes.
    """
    crc_match = crc(decoded_bytes) == 0
    assert crc_match, "CRC check failed."
    return decoded_bytes[:-3]

