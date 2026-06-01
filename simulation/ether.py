import numpy as np

class RFEther:
    """
    Simulates a 32-channel radio spectrum environment.
    Provides methods to step the simulation forward, calculate Signal-to-Noise Ratio (SNR),
    and determine packet collision/delivery metrics.
    """
    def __init__(self, num_channels=32, signal_power=20.0, jammer_power=30.0, noise_floor=1.0, snr_threshold_db=5.0):
        self.num_channels = num_channels
        self.signal_power = signal_power
        self.jammer_power = jammer_power
        self.noise_floor = noise_floor
        # Convert dB threshold to linear: SNR_linear = 10^(SNR_db / 10)
        self.snr_threshold_linear = 10.0 ** (snr_threshold_db / 10.0)
        
        # Reset environmental state
        self.reset()

    def reset(self):
        """Resets the state of the ether."""
        # Initialize background noise (with slight Gaussian variation around noise floor)
        self.channel_noise = np.full(self.num_channels, self.noise_floor)
        self.current_step = 0
        return self.channel_noise

    def step(self, tx_channel, jam_channels):
        """
        Steps the simulation forward by one time step.
        
        Parameters:
            tx_channel (int or None): The channel chosen by the transmitter (0 to num_channels-1).
                                      If None, no transmission occurs.
            jam_channels (list of int): List of channels targeted by the jammer.
            
        Returns:
            dict: A dictionary containing:
                - 'channel_states': 32-element numpy array showing active noise levels.
                - 'jammed_vector': 32-element binary array (1.0 if jammed, 0.0 otherwise).
                - 'snr': SNR value at tx_channel (None if no transmission).
                - 'success': boolean indicating if transmission was successful.
        """
        # 1. Reset channel noise to baseline
        noise = np.full(self.num_channels, self.noise_floor)
        
        # 2. Add some minor random thermal noise fluctuation (e.g. standard deviation 0.05)
        thermal_noise = np.random.normal(0, 0.05, self.num_channels)
        noise = np.clip(noise + thermal_noise, 0.1, None) # keep positive
        
        # 3. Add jammer power to target channels
        jammed_vector = np.zeros(self.num_channels)
        for ch in jam_channels:
            if 0 <= ch < self.num_channels:
                noise[ch] += self.jammer_power
                jammed_vector[ch] = 1.0

        # 4. Calculate transmission success if transmitter active
        success = False
        snr = None
        
        if tx_channel is not None and 0 <= tx_channel < self.num_channels:
            total_noise = noise[tx_channel]
            snr = self.signal_power / total_noise
            
            # Successful if SNR meets threshold
            if snr >= self.snr_threshold_linear:
                success = True
        
        self.current_step += 1
        
        return {
            'channel_states': noise,
            'jammed_vector': jammed_vector,
            'snr': snr,
            'success': success
        }
