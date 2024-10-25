import time
import queue
import copy


class Controller:
    def __init__(self, name, actor_system, state_can, N_FFT, N_CP, N_CP_extra, Fs):
        """
        Initializes the Controller instance for managing state and controlling data dispatch.

        Args:
            name (str): Controller identifier.
            actor_system (object): Reference to the actor system.
            state_can (list): List of available states.
            N_FFT (int): FFT size for processing.
            N_CP (int): Cyclic prefix length.
            N_CP_extra (int): Extra cyclic prefix length for specific symbols.
            Fs (int): Sampling frequency.
        """
        self.name = name
        self.message_queue = queue.Queue()
        self.actor_system = actor_system
        
        self.state_can = state_can
        self.state = state_can[0]  # Initial state
        
        self.read_pointer_in_buffer = 0
        self.write_pointer_in_buffer = 0
        self.n_data_in_buffer = 0  # Available data in buffer
        self.buffer_size_in_dispatcher = None
        self.read_start = 0
        
        self.N_FFT = N_FFT
        self.N_CP = N_CP
        self.pss_start = None
        self.N_id = None
        self.fd = None
        self.ant = 2
        self.N_rb = None
        self.SFN = None
        self.C_phich = None
        
        self.one_frame_len = Fs // 100
        self.one_subframe_len = Fs // 1000
        self.one_slot_len = Fs // 2000
        self.pbch_len = 4 * (N_FFT + N_CP) + N_CP_extra
        self.prior_len = 20
        self.tracking_len = 2 * (N_FFT + N_CP + self.prior_len)
        self.pss_tracking_start = 2 * self.one_frame_len - self.tracking_len // 2
                
        self.counter = 0  # Tracks decoded messages
        self.move_lock = False  # Lock for controlling pointer movement
        self.pss_start_max = 0
        self.first_search = True
        
        self.stop_processing = False

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
        """Processes messages and manages state transitions."""
        # Retrieve the next message
        message = self.message_queue.get()  
        
        # Dispatcher message for updating write pointer and available data
        if message['source'] == 'Dispatcher':
            self.update_buffer(message=message)

            # In 'CellSearch' state, keep sending read control info to Dispatcher for initial cell search
            if self.state == 'CellSearch':
                data_len = (self.n_data_in_buffer - self.read_start) // self.N_FFT * self.N_FFT
                control_info = {
                    'type': 'CR', 
                    'destination': 'CellSearch', 
                    'source': 'Controller', 
                    'start': self.read_start,
                    'len': data_len
                }
                self.actor_system.send_message('Dispatcher', control_info)
                self.read_start += data_len

        # Completion of cell search or tracking triggers MIB decode state
        elif (message['source'] == 'CellSearch' and self.first_search) or message['source'] == 'FrameTrack':
            if message['source'] == 'CellSearch':
                print('Initial Cell Search Done.')
            elif message['source'] == 'FrameTrack':
                print('One-time Frame Tracking Done.')
                
            self.first_search = False  # initial Cell search no longer needed
            self.update_sys(message)

            if self.state == 'CellSearch':
                self.pss_start = message['pss_start']
                self.update_state('CellSearch')
                tmp_pss_start = self.pss_start
            elif self.state == 'MIBDecode':  # Adjust start for even frame if necessary
                self.pss_start += self.one_frame_len + message['pss_start'] - self.tracking_len // 2
                tmp_pss_start = self.pss_start
            elif self.state == 'SIB1Decode':
                self.pss_start = message['start'] + message['pss_start']
                tmp_pss_start = self.pss_start + self.read_pointer_in_buffer

            if self.pss_start_max < self.pss_start:
                self.pss_start_max = self.pss_start

            print(tmp_pss_start, self.N_id, self.fd)

            # Send control info for MIB decoding
            control_info = {
                'type': 'CR', 
                'destination': 'MIBDecode', 
                'source': 'Controller',
                'start': self.pss_start + self.N_FFT, 
                'len': self.pbch_len,
                'N_id': self.N_id, 
                'fd': self.fd, 
                'ant': self.ant, 
                'h': []
            }
            # if sufficient data are stored in the circular buffer
            if self.pss_start + self.N_FFT + self.pbch_len <= self.n_data_in_buffer:
                self.actor_system.send_message('Dispatcher', control_info) 
            else: # if not, delay and re-send this information to the controller 
                time.sleep(2)
                self.actor_system.send_message('Controller', control_info)

        # Transition to SIB1 decode upon MIB decoding completion
        elif message['source'] == 'MIBDecode':
            self.update_sys(message)

            # Odd SFN triggers FrameTrack for PSS at the next (even) frame; otherwise, proceed with SIB1 decode
            if self.SFN % 2:
                next_pss_start = self.pss_start + self.one_frame_len
                start = next_pss_start - self.tracking_len // 2
                control_info = {
                    'type': 'CR', 
                    'destination': 'FrameTrack', 
                    'source': 'Controller', 
                    'start': start, 
                    'len': self.tracking_len
                }
                # if sufficient data are stored in the circular buffer
                if start + self.tracking_len <= self.n_data_in_buffer:
                    self.actor_system.send_message('Dispatcher', control_info)
                else: # if not, delay and re-send this information to the controller
                    time.sleep(2)
                    self.actor_system.send_message('Controller', control_info)
            else:
                print('MIB Decoding Done.')
                print(self.N_rb, self.SFN, self.C_phich)
                
                if self.state == 'MIBDecode':
                    self.update_state('MIBDecode')

                sib1_subframe_start = self.pss_start + self.N_FFT + int(4.5 * self.one_subframe_len)
                control_info = {
                    'type': 'CR', 
                    'destination': 'SIB1Decode', 
                    'source': 'Controller',
                    'start': sib1_subframe_start, 
                    'len': self.one_subframe_len,
                    'N_rb': self.N_rb, 
                    'N_id': self.N_id, 
                    'fd': self.fd, 
                    'ant': self.ant,
                    'h': [], 
                    'C_phich': self.C_phich
                }
                # if sufficient data are stored in the circular buffer
                if sib1_subframe_start + self.one_subframe_len <= self.n_data_in_buffer:
                    self.actor_system.send_message('Dispatcher', control_info)
                else: # if not, delay and re-send this information to the controller
                    time.sleep(2)
                    self.actor_system.send_message('Controller', control_info)

        # Handling periodic SIB1 decoding and updating pointers as needed
        elif message['source'] == 'SIB1Decode':
            print('SIB1 Decoding Done.')
            print(message['sib1_info'])

            # for frametracking: multiple read info can be sent simultaneously.
            if self.counter > 0:
                self.counter -= 1

            if self.counter == 0 and not self.move_lock:
                print('All Sent SIB1s Have Been Decoded, Move Read Pointer.')
                self.update_buffer(move_len=self.pss_start_max)
                control_info = {
                    'type': 'CM', # move read pointer
                    'destination': 'Dispatcher', 
                    'source': 'Controller', 
                    'len': self.pss_start_max
                }
                self.actor_system.send_message('Dispatcher', control_info)
                self.move_lock = True
                self.pss_start_max = 0  # Reset max start position after move

            control_info = {
                'type': 'CR', 
                'destination': 'FrameTrack', 
                'source': 'Controller', 
                'start': self.pss_tracking_start, 
                'len': self.tracking_len
            }
            if self.pss_tracking_start + self.tracking_len > self.n_data_in_buffer:
                time.sleep(2)
                self.actor_system.send_message('Controller', control_info)
            else:
                self.move_lock = False # if frame tracking starts, release move lock.
                pss_tracking_tmp = self.pss_tracking_start
                while pss_tracking_tmp + self.tracking_len <= self.n_data_in_buffer:
                    self.counter += 1
                    control_info['start'] = pss_tracking_tmp % self.buffer_size_in_dispatcher
                    self.actor_system.send_message('Dispatcher', copy.deepcopy(control_info))
                    pss_tracking_tmp += self.one_frame_len * 2

        # Handling requeued messages if insufficient data in buffer
        elif message['source'] == 'Controller':
            print('Processing Requeued Message ...')
            if message['start'] + message['len'] > self.n_data_in_buffer:
                time.sleep(2)
                self.actor_system.send_message('Controller', message)
            else:
                if message['destination'] == 'FrameTrack' and self.state == 'SIB1Decode':
                    pss_tracking_tmp = message['start']
                    self.move_lock = False
                    while pss_tracking_tmp + message['len'] <= self.n_data_in_buffer:
                        self.counter += 1
                        message['start'] = pss_tracking_tmp % self.buffer_size_in_dispatcher
                        self.actor_system.send_message('Dispatcher', copy.deepcopy(message))
                        pss_tracking_tmp += self.one_frame_len * 2
                else:
                    self.actor_system.send_message('Dispatcher', message)

    def update_buffer(self, message=None, move_len=None):
        """
        Updates buffer pointers and available data based on write/move length.
        """
        if message:
            if self.buffer_size_in_dispatcher != message['buffer_size']:
                self.buffer_size_in_dispatcher = message['buffer_size']
            self.write_pointer_in_buffer = message['write_p']
        if move_len:
            self.read_pointer_in_buffer = (self.read_pointer_in_buffer + move_len) % self.buffer_size_in_dispatcher
        self.n_data_in_buffer = (self.write_pointer_in_buffer - self.read_pointer_in_buffer + self.buffer_size_in_dispatcher) % self.buffer_size_in_dispatcher

    def update_state(self, state):
        """
        Advances to the next state if it exists in state_can.
        """
        if state in self.state_can:
            self.state = self.state_can[self.state_can.index(state) + 1]
        else:
            print('No state recorded in the state can')

    def update_sys(self, message):
        """
        Updates system parameters based on the message source.
        """
        if message['source'] in ['CellSearch', 'FrameTrack']:
            self.N_id, self.fd = message['N_id'], message['fd']
        if message['source'] == 'MIBDecode':
            self.N_rb, self.SFN, self.C_phich = message['N_rb'], message['SFN'], message['C_phich']
            
            

