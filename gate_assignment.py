import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import time
import streamlit as st

def convert_time_to_minutes(time_str):
    """Convert time in HHMM format to minutes since midnight"""
    hours = int(time_str) // 100
    minutes = int(time_str) % 100
    return hours * 60 + minutes

class Gate:
    def __init__(self, number, distance):
        self.number = number
        self.distance = distance
        self.occupied_until = None
        self.current_flight = None
    
    def is_available_at(self, time):
        if self.occupied_until is None:
            return True
        return time >= self.occupied_until

class GateAssignment:
    def __init__(self, gates_file):
        # Read gate distances from CSV
        gates_df = pd.read_csv(gates_file)
        # Create gates sorted by distance from runway
        self.gates = [
            Gate(row['gate_number'], row['distance_from_runway'])
            for _, row in gates_df.sort_values('distance_from_runway').iterrows()
        ]
        self.waiting_queue = deque()
        
    def find_closest_available_gate(self, arrival_time):
        """Find the closest available gate to runway"""
        for gate in self.gates:  # Gates are already sorted by distance
            if gate.is_available_at(arrival_time):
                return gate
        return None
    
    def process_waiting_queue(self, current_time):
        """Try to assign gates to waiting flights"""
        assignments = []
        remaining_queue = deque()
        
        while self.waiting_queue:
            waiting_flight = self.waiting_queue.popleft()
            gate = self.find_closest_available_gate(current_time)
            
            if gate:
                # Gate found for waiting flight
                gate.occupied_until = current_time + 60  # 60 minutes occupancy
                gate.current_flight = waiting_flight
                assignments.append({
                    'flight_time': waiting_flight['CRS_DEP_TIME'],
                    'pred_delay': waiting_flight['pred_delay'],
                    'arrival_time': waiting_flight['CRS_ARR_TIME'],
                    'actual_arrival_time': current_time,
                    'gate_number': gate.number,
                    'gate_distance': gate.distance,
                    'wait_time': current_time - waiting_flight['arrival_time']
                })
            else:
                # Still no gate available, put back in queue
                remaining_queue.append(waiting_flight)
        
        self.waiting_queue = remaining_queue
        return assignments
    
    def assign_gates(self, flights_df):
        # Convert times to minutes for easier comparison
        flights_df['actual_departure'] = flights_df.apply(
            lambda x: convert_time_to_minutes(x['CRS_DEP_TIME']) + x['DEP_DELAY'],
            axis=1
        )
        
        # Calculate arrival time in minutes
        flights_df['arrival_time'] = flights_df.apply(
            lambda x: convert_time_to_minutes(x['CRS_ARR_TIME']) + x['ARR_DELAY'],
            axis=1
        )
        
        assignments = []
        current_time = flights_df['arrival_time'].min()
        max_time = flights_df['arrival_time'].max() + 120  # Extra time to process queue
        
        while current_time <= max_time:
            # Process any waiting flights first
            assignments.extend(self.process_waiting_queue(current_time))
            
            # Process new arrivals at current time
            current_flights = flights_df[flights_df['arrival_time'] == current_time]
            
            for _, flight in current_flights.iterrows():
                gate = self.find_closest_available_gate(current_time)
                
                if gate:
                    gate.occupied_until = current_time + 60  # 60 minutes occupancy
                    gate.current_flight = flight
                    assignments.append({
                        'flight_time': flight['CRS_DEP_TIME'],
                        'pred_delay': flight['pred_delay'],
                        'arrival_time': flight['CRS_ARR_TIME'],
                        'actual_arrival_time': current_time,
                        'gate_number': gate.number,
                        'gate_distance': gate.distance,
                        'wait_time': 0
                    })
                else:
                    # Add to waiting queue if no gate available
                    self.waiting_queue.append(flight)
            
            current_time += 1
        
        # Add any remaining queued flights as unassigned
        for flight in self.waiting_queue:
            assignments.append({
                'flight_time': flight['CRS_DEP_TIME'],
                'pred_delay': flight['pred_delay'],
                'arrival_time': flight['CRS_ARR_TIME'],
                'actual_arrival_time': None,
                'gate_number': None,
                'gate_distance': None,
                'wait_time': None
            })
        
        # Create a dictionary to store gate assignment details
        gate_assignments = {}
        current_time = pd.Timestamp.now()
        
        for idx, flight in flights_df.iterrows():
            wait_time = 0
            assigned_gate = None
            
            # Store the assignment details
            gate_assignments[flight['flight_number']] = {
                'gate': assigned_gate,
                'assignment_time': current_time,
                'wait_time': wait_time
            }
        
        # Store in session state
        st.session_state['gate_assignments'] = gate_assignments
        return gate_assignments

    def assign_gate(self, flight):
        start_time = time.time()
        # Find an available gate
        assigned_gate = self.find_closest_available_gate(flight['arrival_time'])
        
        wait_time = int((time.time() - start_time) * 60)  # Convert to minutes
        return assigned_gate, wait_time

def main():
    # Read the preprocessed CSV file
    df = pd.read_csv('preprocessed_flights.csv')
    
    # Create gate assignment instance with gate distances
    gate_manager = GateAssignment('gate_distances.csv')
    
    # Assign gates
    assignments = gate_manager.assign_gates(df)
    
    # Display results
    print("\nGate Assignments:")
    print(assignments.to_string(index=False))
    
    # Print statistics
    unassigned = assignments['gate_number'].isna().sum()
    total_wait_time = assignments['wait_time'].sum()
    max_wait_time = assignments['wait_time'].max()
    
    print(f"\nStatistics:")
    print(f"Total unassigned flights: {unassigned}")
    print(f"Total wait time (minutes): {total_wait_time}")
    print(f"Maximum wait time (minutes): {max_wait_time}")
    
    if unassigned > 0:
        print(f"\nWarning: {unassigned} flights could not be assigned to gates")

if __name__ == "__main__":
    main() 