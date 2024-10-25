import queue
import numpy as np
import time


class Dispatcher:
    def __init__(self, name, actor_system, buffer_size, N_FFT, N_CP, N_CP_extra):
        """
        Initializes a Dispatcher instance to handle circular buffer and message processing.

        Args:
            name (str): Identifier for the dispatcher.
            actor_system (object): Reference to the actor system.
            buffer_size (int): Size of the internal circular buffer.
            N_FFT (int): FFT size for processing.
            N_CP (int): Cyclic prefix length.
            N_CP_extra (int): Additional cyclic prefix for specific symbols.
        """
        self.name = name
        self.message_queue = queue.Queue()  # FIFO message queue for received messages
        self.actor_system = actor_system
        
        self.N_CP = N_CP                  # Regular cyclic prefix
        self.N_CP_extra = N_CP_extra      # Extra cyclic prefix for specific symbols
        self.N_FFT = N_FFT
        self.circular_buffer = CircularBuffer(buffer_size)  # Internal circular buffer
        
        self.counter = 0                  # Track messages processed
        self.send_timer = 16              # Frequency of control info sending
        self.stop_processing = False      # Flag to halt processing
        
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
        Processes messages from the queue, handling data writes and reads based on message type.
        """
        # Retrieve next message
        message = self.message_queue.get()  

        # Write data to circular buffer
        if message['type'] == 'W':

            self.circular_buffer.write_data(message['data'])
            self.counter += 1

            # Send control message after every set interval
            if self.counter == self.send_timer:
                control_info = {
                    'type': 'C', 
                    'destination': 'Controller', 
                    'source': 'Dispatcher',
                    'read_p': self.circular_buffer.read_pointer, 
                    'write_p': self.circular_buffer.write_pointer,
                    'buffer_size': self.circular_buffer.buffer_size
                }
                self.actor_system.send_message('Controller', control_info)
                self.counter = 0

        # Handle control read messages
        elif message['type'] == 'CR':
            data = self.circular_buffer.read_data(message['start'], message['len'])
            data = np.array(data)

            # Dispatch messages based on destination
            if message['destination'] == 'CellSearch':
                for n in range(message['len'] // self.N_FFT):
                    tmp_data = data[n * self.N_FFT:(n + 1) * self.N_FFT]
                    data_info = {
                        'type': 'D', 
                        'destination': 'CellSearch', 
                        'source': 'Dispatcher',
                        'data': tmp_data
                    }
                    self.actor_system.send_message('CellSearch', data_info)
                    time.sleep(0.05)

            elif message['destination'] == 'MIBDecode':
                start = 0
                for l in range(4):
                    CP = self.N_CP + self.N_CP_extra if l == 0 else self.N_CP
                    tmp_data = data[start:start + self.N_FFT + CP]
                    start += self.N_FFT + CP
                    data_info = {
                        'type': 'D', 
                        'destination': 'MIBDecode', 
                        'source': 'Dispatcher',
                        'data': tmp_data, 
                        'ns': 1, 
                        'l': l, 
                        'N_rb': 6,
                        'N_id': message['N_id'], 
                        'fd': message['fd'], 
                        'ant': message['ant']
                    }
                    self.actor_system.send_message('ChannelEstimate', data_info)
                    time.sleep(0.05)

            elif message['destination'] == 'SIB1Decode':
                start = 0
                ns = 10
                for l in range(14):
                    if l >= 7:
                        ns = 11
                        l %= 7
                    CP = self.N_CP + self.N_CP_extra if l == 0 else self.N_CP
                    tmp_data = data[start:start + self.N_FFT + CP]
                    start += self.N_FFT + CP
                    data_info = {
                        'type': 'D', 
                        'destination': 'SIB1Decode', 
                        'source': 'Dispatcher',
                        'data': tmp_data, 
                        'ns': ns, 
                        'l': l, 
                        'N_rb': message['N_rb'],
                        'N_id': message['N_id'], 
                        'fd': message['fd'], 
                        'ant': message['ant']
                    }
                    self.actor_system.send_message('ChannelEstimate', data_info)
                    time.sleep(0.05)

            elif message['destination'] == 'FrameTrack':
                data_info = {
                    'type': 'D', 
                    'destination': 'FrameTrack', 
                    'source': 'Dispatcher',
                    'data': data, 
                    'start': message['start']
                }
                self.actor_system.send_message('CellSearch', data_info)

        # Move read pointer based on control message
        elif message['type'] == 'CM':
            self.circular_buffer.move_read(message['len'])



class CircularBuffer:
    def __init__(self, buffer_size):
        """
        Initializes a circular buffer with a specific size and manages read/write pointers.

        Args:
            buffer_size (int): Maximum size of the buffer.
        """
        self.read_pointer = 0          # Tracks where to read data from
        self.write_pointer = 0         # Tracks where to write data to
        self.pass_origin = False       # Tracks if write pointer has looped past read pointer
        self.buffer_size = buffer_size # Maximum buffer capacity
        self.buffer = [None] * buffer_size  # Initialize buffer with None values
        self.blocking = False          # Indicates if buffer is in a blocking state for writes

    def read_data(self, start, length):
        """
        Reads a specified length of data starting from a given position.

        Args:
            start (int): Offset from the current read pointer.
            length (int): Number of elements to read.

        Returns:
            Array of data or None if there's insufficient data.
        """
        start += self.read_pointer  # Adjust start relative to read pointer
        read_end = start + length

        if length > self.buffer_size:
            print("Warning: Requested read length exceeds buffer size.")
            return None
        
        # Case 1: Non-overflow, enough data between read and write pointers
        if not self.pass_origin:
            if read_end <= self.write_pointer:
                return self.buffer[start:read_end]
            else:
                print("Warning: Insufficient data available to read.")
        else:
            # Case 2: Handle overflow with wrap-around
            if read_end <= self.buffer_size:
                return self.buffer[start:read_end]
            elif read_end % self.buffer_size <= self.write_pointer:
                if start <= self.buffer_size:
                    return np.concatenate((self.buffer[start:], self.buffer[:read_end % self.buffer_size]))
                else:
                    return self.buffer[start % self.buffer_size:read_end % self.buffer_size]
            else:
                print("Warning: Insufficient data available to read.")

    def write_data(self, data):
        """
        Writes data to the buffer, respecting buffer capacity and wrap-around logic.

        Args:
            data (array-like): Data to be written into the buffer.
        """
        write_end = self.write_pointer + data.size

        # Case 1: Non-overflow, sufficient space for data
        if not self.pass_origin:
            if write_end < self.buffer_size:
                self.buffer[self.write_pointer:write_end] = data
                self.write_pointer = write_end
            elif write_end % self.buffer_size <= self.read_pointer:
                # Wrap-around without exceeding read pointer
                self.pass_origin = True
                self.buffer[self.write_pointer:] = data[:self.buffer_size - self.write_pointer]
                self.buffer[:write_end % self.buffer_size] = data[self.buffer_size - self.write_pointer:]
                self.write_pointer = write_end % self.buffer_size
            else:
                print("Warning: Insufficient buffer space to write data.")
                self.blocking = True
        elif write_end <= self.read_pointer:
            # Case 2: Handle wrap-around with enough room before read pointer
            self.buffer[self.write_pointer:write_end] = data
            self.write_pointer = write_end
        else:
            print("Warning: Insufficient buffer space to write data.")
            self.blocking = True

    def move_read(self, length):
        """
        Advances the read pointer by a specified length.

        Args:
            length (int): Number of elements to move the read pointer by.
        """
        new_read_pointer = self.read_pointer + length
        if not self.pass_origin:
            if new_read_pointer <= self.write_pointer:
                self.read_pointer = new_read_pointer
            else:
                print("Warning: Read pointer cannot advance past write pointer.")
        else:
            if new_read_pointer <= self.buffer_size:
                self.read_pointer = new_read_pointer
            elif new_read_pointer % self.buffer_size <= self.write_pointer:
                self.read_pointer = new_read_pointer % self.buffer_size
                self.pass_origin = False
            else:
                print("Warning: Read pointer cannot advance past write pointer.")

    def reset_read(self):
        """
        Resets the read pointer to the start of the buffer.
        """
        self.read_pointer = 0

    def reset_write(self):
        """
        Resets the write pointer to the start of the buffer.
        """
        self.write_pointer = 0

    def reset_blocking(self):
        """
        Clears the blocking status of the buffer.
        """
        self.blocking = False


