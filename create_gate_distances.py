import pandas as pd

# Create data for 20 gates with their distances from runway
gates_data = {
    'gate_number': list(range(20)),
    # Distances in meters - assuming gates are arranged in a way that lower numbered gates
    # are closer to the runway
    'distance_from_runway': [
        50, 60, 75, 90, 100,    # Gates 0-4: Very close
        120, 140, 160, 180, 200, # Gates 5-9: Moderately close
        250, 275, 300, 325, 350, # Gates 10-14: Medium distance
        400, 450, 500, 550, 600  # Gates 15-19: Further away
    ]
}

# Create DataFrame and save to CSV
gates_df = pd.DataFrame(gates_data)
gates_df.to_csv('data/gate_distances.csv', index=False)
print("Gate distances file created successfully") 