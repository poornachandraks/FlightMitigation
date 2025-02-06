import pandas as pd

def preprocess_flight_data(input_file, output_file):
    """
    Preprocess the flight data:
    1. Read the CSV file
    2. Calculate actual arrival time
    3. Sort by actual arrival time and predicted delay
    4. Save the preprocessed data
    """
    # Read the original CSV file
    df = pd.read_csv(input_file)
    
    # Convert times to numeric if they aren't already
    df['CRS_ARR_TIME'] = pd.to_numeric(df['CRS_ARR_TIME'])
    df['ARR_DELAY'] = pd.to_numeric(df['ARR_DELAY'])
    
    # Calculate actual arrival time (scheduled + delay)
    df['ACTUAL_ARR_TIME'] = df['CRS_ARR_TIME'] + df['ARR_DELAY']
    
    # Sort the dataframe by actual arrival time and then by predicted delay
    df_sorted = df.sort_values(
        ['ACTUAL_ARR_TIME', 'pred_delay'], 
        ascending=[True, False]  # True for arrival time, False for pred_delay
    )
    
    # Save the preprocessed data
    df_sorted.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total flights: {len(df_sorted)}")
    print("\nArrival Time Statistics:")
    print(f"Scheduled time range: {df_sorted['CRS_ARR_TIME'].min()} to {df_sorted['CRS_ARR_TIME'].max()}")
    print(f"Actual time range: {df_sorted['ACTUAL_ARR_TIME'].min()} to {df_sorted['ACTUAL_ARR_TIME'].max()}")
    print("\nDelay Statistics:")
    print(f"Average arrival delay: {df_sorted['ARR_DELAY'].mean():.2f} minutes")
    print(f"Maximum arrival delay: {df_sorted['ARR_DELAY'].max():.2f} minutes")
    print(f"Average predicted delay: {df_sorted['pred_delay'].mean():.2f} minutes")
    print(f"Maximum predicted delay: {df_sorted['pred_delay'].max():.2f} minutes")

if __name__ == "__main__":
    input_file = 'data/delay_data.csv'
    output_file = 'data/preprocessed_flights.csv'
    preprocess_flight_data(input_file, output_file) 