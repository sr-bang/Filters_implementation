class BayesFilter:
    def __init__(self):
        # Initial state probabilities
        self.initial_belief_open = 0.5
        self.initial_belief_closed = 0.5

        # Transition probabilities for actions: 0 = do_nothing, 1 = open
        self.transition_probs = {
            0: {'open': {'open': 1, 'closed': 0}, 'closed': {'open': 0, 'closed': 1}},
            1: {'open': {'open': 1, 'closed': 0}, 'closed': {'open': 0.8, 'closed': 0.2}}
        }

        # Sensor probabilities for measurements: 1 = open, 0 = closed
        self.sensor_probs = {
            1: {'open': 0.6, 'closed': 0.2},
            0: {'open': 0.4, 'closed': 0.8}
        }
    
    # Predict the next state given the current state and action
    def predict(self, action, belief_open, belief_closed):
        '''
        action: the action taken
        belief_open: the belief that the door is open
        belief_closed: the belief that the door is closed
        return: the predicted belief that the door is open and closed
        '''
        trans = self.transition_probs[action]
        predicted_open = trans['open']['open'] * belief_open + trans['closed']['open'] * belief_closed
        # predicted_closed = trans['open']['closed'] * belief_open + trans['closed']['closed'] * belief_closed
        predicted_closed = 1 - predicted_open
        return predicted_open, predicted_closed
    
    # Update the belief given the measurement
    def update(self, measurement, predicted_open, predicted_closed):
        '''
        measurement: the measurement taken
        predicted_open: the predicted belief that the door is open
        predicted_closed: the predicted belief that the door is closed
        return: the updated belief that the door is open and closed
        '''
        sensor = self.sensor_probs[measurement]
        updated_open = sensor['open'] * predicted_open
        updated_closed = sensor['closed'] * predicted_closed
        # Normalize the probabilities
        normalization_factor = 1 / (updated_open + updated_closed)
        updated_open *= normalization_factor
        updated_closed = 1 - updated_open
        return updated_open, updated_closed
    
    # Run the filter until the belief that the door is open is above a threshold
    def run_filter(self, action, measurement, threshold):
        '''
        action: the action taken
        measurement: the measurement taken
        threshold: the threshold for the belief that the door is open
        return: the predicted belief that the door is open and closed, and the number of iterations
        '''
        belief_open, belief_closed = self.initial_belief_open, self.initial_belief_closed
        i = 0
        while True:
            # Predict the next state
            predicted_open, predicted_closed = self.predict(action, belief_open, belief_closed)
            # Update the belief given the measurement
            belief_open, belief_closed = self.update(measurement, predicted_open, predicted_closed)
            i += 1
            if belief_open >= threshold:
                break
        return predicted_open, predicted_closed, i
    

# Test the filter

bayesian_filter = BayesFilter()

# Task 1. Action: do nothing (0), Measurement: door open (1)
action = 0
measurement = 1
threshold = 0.9999
predicted_open, predicted_closed, iterations = bayesian_filter.run_filter(action, measurement, threshold)
print(f"Iterations for 'do nothing' and 'door open': {iterations}")

# Task 2. Action: push (1), Measurement: door open (1)
action = 1
predicted_open, predicted_closed, iterations = bayesian_filter.run_filter(action, measurement, threshold)
print(f"Iterations for 'push' and 'door open': {iterations}")

# Task 3. Action: push (1), Measurement: door closed (0)
measurement = 0
predicted_open, predicted_closed, iterations = bayesian_filter.run_filter(action, measurement, threshold)
print(f"Steady state belief for 'push' and 'door closed': open={predicted_open}, closed={predicted_closed}, Iterations: {iterations}")