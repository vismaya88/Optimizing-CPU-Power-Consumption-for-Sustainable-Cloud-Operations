import pandas as pd
import matplotlib.pyplot as plt
import psutil
import time

file_path = "15.csv"  
df = pd.read_csv(file_path)

MIN_FREQ_MULTIPLIER = 0.5  
MAX_FREQ_MULTIPLIER = 1.2  
C = 1  

mean_cpu_usage = df['CPU usage [%]'].mean()
std_cpu_usage = df['CPU usage [%]'].std()
CPU_HIGH_THRESHOLD = mean_cpu_usage + std_cpu_usage
CPU_LOW_THRESHOLD = mean_cpu_usage - std_cpu_usage

def estimate_power_consumption(frequency_mhz):
    return C * (frequency_mhz)

def dvfs_algorithm(cpu_usage_percent, current_freq, cpu_max_freq):
    if cpu_usage_percent > CPU_HIGH_THRESHOLD:
        new_freq = min(current_freq * MAX_FREQ_MULTIPLIER, cpu_max_freq)
        action = "Scaling up"
    elif cpu_usage_percent < CPU_LOW_THRESHOLD:
        new_freq = max(current_freq * MIN_FREQ_MULTIPLIER, cpu_max_freq * 0.5)
        action = "Scaling down"
    else:
        new_freq = current_freq
        action = "No change"
    
    return new_freq, action, estimate_power_consumption(new_freq)

def dpm_algorithm(cpu_usage_percent, current_freq):
    if cpu_usage_percent < CPU_LOW_THRESHOLD:
        new_freq = current_freq * 0.7  
        action = "Idle/Reduced activity"
    else:
        new_freq = current_freq
        action = "Normal operation"
    
    return new_freq, action, estimate_power_consumption(new_freq)

def avs_algorithm(cpu_usage_percent, current_freq, cpu_max_freq):
    if cpu_usage_percent > CPU_HIGH_THRESHOLD:
        new_freq = min(current_freq * MAX_FREQ_MULTIPLIER, cpu_max_freq)
        action = "Scaling up"
    elif cpu_usage_percent < CPU_LOW_THRESHOLD:
        new_freq = max(current_freq * MIN_FREQ_MULTIPLIER, cpu_max_freq * 0.5)
        action = "Scaling down"
    else:
        new_freq = current_freq
        action = "No change"
    
    voltage_scaling_factor = 0.9  
    new_power = estimate_power_consumption(new_freq) * voltage_scaling_factor ** 2
    return new_freq, action, new_power

df['New CPU frequency [MHz]'] = None
df['DVFS Action'] = None
df['Power without DVFS'] = df['CPU capacity provisioned [MHZ]'].apply(estimate_power_consumption)
df['Power with DVFS'] = None
df['DPM New Frequency [MHz]'] = None
df['DPM Action'] = None
df['DPM Power with'] = None
df['AVS New Frequency [MHz]'] = None
df['AVS Action'] = None
df['AVS Power with'] = None

initial_freq = df.iloc[0]['CPU capacity provisioned [MHZ]']  

for i, row in df.iterrows():
    # DVFS
    new_freq, action, power_with_dvfs = dvfs_algorithm(row['CPU usage [%]'], initial_freq, row['CPU capacity provisioned [MHZ]'])
    df.at[i, 'New CPU frequency [MHz]'] = new_freq
    df.at[i, 'DVFS Action'] = action
    df.at[i, 'Power with DVFS'] = power_with_dvfs
    
    # DPM
    new_freq_dpm, action_dpm, power_with_dpm = dpm_algorithm(row['CPU usage [%]'], initial_freq)
    df.at[i, 'DPM New Frequency [MHz]'] = new_freq_dpm
    df.at[i, 'DPM Action'] = action_dpm
    df.at[i, 'DPM Power with'] = power_with_dpm
    
    # AVS
    new_freq_avs, action_avs, power_with_avs = avs_algorithm(row['CPU usage [%]'], initial_freq, row['CPU capacity provisioned [MHZ]'])
    df.at[i, 'AVS New Frequency [MHz]'] = new_freq_avs
    df.at[i, 'AVS Action'] = action_avs
    df.at[i, 'AVS Power with'] = power_with_avs
    
    initial_freq = new_freq  

output_file_historical = "historical_output_15.csv"

cpu_max_freq = psutil.cpu_freq().max  
iterations = 10  

real_time_df = pd.DataFrame(columns=['Timestamp', 'CPU usage [%]', 'Current Frequency [MHz]',
                                     'New Frequency [MHz]', 'DVFS Action', 'DPM Action', 'AVS Action',
                                     'Power without DVFS', 'Power with DVFS', 'Power with DPM', 'Power with AVS'])
df.rename(columns={'DPM Power with': 'Power with DPM'}, inplace=True)
df.rename(columns={'AVS Power with': 'Power with AVS'}, inplace=True)
real_time_df.rename(columns={'CPU Usage': 'CPU usage [%]'}, inplace=True)


for i in range(iterations):
    cpu_usage_percent = psutil.cpu_percent(interval=1)  
    current_freq = psutil.cpu_freq().current  

    # Apply DVFS
    new_freq, action, power_with_dvfs = dvfs_algorithm(cpu_usage_percent, current_freq, cpu_max_freq)
    
    # Apply DPM
    new_freq_dpm, action_dpm, power_with_dpm = dpm_algorithm(cpu_usage_percent, current_freq)
    
    # Apply AVS
    new_freq_avs, action_avs, power_with_avs = avs_algorithm(cpu_usage_percent, current_freq, cpu_max_freq)
    
    # Power without DVFS (assuming max frequency)
    power_no_dvfs = estimate_power_consumption(cpu_max_freq)
    
    real_time_df = pd.concat([real_time_df, pd.DataFrame({
    'Timestamp': [time.time()],
    'CPU Usage [%]': [cpu_usage_percent],
    'Current Frequency [MHz]': [current_freq],
    'New Frequency [MHz]': [new_freq],
    'DVFS Action': [action],
    'DPM Action': [action_dpm],
    'AVS Action': [action_avs],
    'Power without DVFS': [power_no_dvfs],
    'Power with DVFS': [power_with_dvfs],
    'Power with DPM': [power_with_dpm],
    'Power with AVS': [power_with_avs]
})], ignore_index=True)

    print(f"Timestamp: {time.time()}, CPU Usage: {cpu_usage_percent:.2f}%, Action (DVFS): {action}, Power with DVFS: {power_with_dvfs:.2f}")

output_file_real_time = "real_time_output.csv"
real_time_df.to_csv(output_file_real_time, index=False)

print(f"Historical data with DVFS, DPM, and AVS saved to {output_file_historical}")
print(f"Real-time DVFS, DPM, and AVS data saved to {output_file_real_time}")

def calculate_performance_matrix(df):
    results = {}

    # Average Power Consumption
    results['Average Power without DVFS'] = df['Power without DVFS'].mean()
    results['Average Power with DVFS'] = df['Power with DVFS'].mean()
    results['Average Power with DPM'] = df['Power with DPM'].mean()
    results['Average Power with AVS'] = df['Power with AVS'].mean()

    # Power Savings
    results['Power Savings (DVFS)'] = results['Average Power without DVFS'] - results['Average Power with DVFS']
    results['Power Savings (DPM)'] = results['Average Power without DVFS'] - results['Average Power with DPM']
    results['Power Savings (AVS)'] = results['Average Power without DVFS'] - results['Average Power with AVS']

    # CPU Utilization Efficiency (CPU usage / Power consumption)
    results['CPU Utilization Efficiency (DVFS)'] = df['CPU usage [%]'].mean() / results['Average Power with DVFS']
    results['CPU Utilization Efficiency (DPM)'] = df['CPU usage [%]'].mean() / results['Average Power with DPM']
    results['CPU Utilization Efficiency (AVS)'] = df['CPU usage [%]'].mean() / results['Average Power with AVS']

    # Energy Efficiency 
    results['Energy Efficiency (DVFS)'] = df['CPU usage [%]'].mean() / results['Average Power with DVFS']
    results['Energy Efficiency (DPM)'] = df['CPU usage [%]'].mean() / results['Average Power with DPM']
    results['Energy Efficiency (AVS)'] = df['CPU usage [%]'].mean() / results['Average Power with AVS']

    return pd.DataFrame(results, index=[0])

performance_matrix = calculate_performance_matrix(df)
print("\nPerformance Matrix:\n", performance_matrix)
performance_matrix_file = "performance_matrix.csv"
performance_matrix.to_csv(performance_matrix_file, index=False)

print(f"Performance matrix saved to {performance_matrix_file}")

def calculate_accuracy(df):
    total_rows = len(df)
    
    dvfs_accuracy = (
        ((df['CPU usage [%]'] > CPU_HIGH_THRESHOLD) & (df['DVFS Action'] == "Scaling up")) |
        ((df['CPU usage [%]'] < CPU_LOW_THRESHOLD) & (df['DVFS Action'] == "Scaling down")) |
        ((df['CPU usage [%]'] >= CPU_LOW_THRESHOLD) & (df['CPU usage [%]'] <= CPU_HIGH_THRESHOLD) & (df['DVFS Action'] == "No change"))
    ).sum()

    dpm_accuracy = (
        ((df['CPU usage [%]'] < CPU_LOW_THRESHOLD) & (df['DPM Action'] == "Idle/Reduced activity")) |
        ((df['CPU usage [%]'] >= CPU_LOW_THRESHOLD) & (df['DPM Action'] == "Normal operation"))
    ).sum()

    avs_accuracy = (
        ((df['CPU usage [%]'] > CPU_HIGH_THRESHOLD) & (df['AVS Action'] == "Scaling up")) |
        ((df['CPU usage [%]'] < CPU_LOW_THRESHOLD) & (df['AVS Action'] == "Scaling down")) |
        ((df['CPU usage [%]'] >= CPU_LOW_THRESHOLD) & (df['CPU usage [%]'] <= CPU_HIGH_THRESHOLD) & (df['AVS Action'] == "No change"))
    ).sum()


    return dvfs_accuracy, dpm_accuracy, avs_accuracy

performance_matrix = calculate_performance_matrix(df)

def plot_power_consumption(df_historical, df_real_time):
    required_columns = ['Power without DVFS', 'Power with DVFS', 'Power with DPM', 'Power with AVS']
    for col in required_columns:
        if col not in df_historical.columns or col not in df_real_time.columns:
            raise ValueError(f"Missing required column '{col}' in historical or real-time DataFrame.")

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Historical data
    axs[0].plot(df_historical.index, df_historical['Power without DVFS'], label='Power without DVFS', color='blue', linestyle='--')
    axs[0].plot(df_historical.index, df_historical['Power with DVFS'], label='Power with DVFS', color='green')
    axs[0].plot(df_historical.index, df_historical['Power with DPM'], label='Power with DPM', color='orange')
    axs[0].plot(df_historical.index, df_historical['Power with AVS'], label='Power with AVS', color='red')
    axs[0].set_title('Historical Data: Power Consumption Comparison')
    axs[0].set_xlabel('Data Points')
    axs[0].set_ylabel('Power Consumption')
    axs[0].legend()
    axs[0].grid()

    # Real-time data
    axs[1].plot(df_real_time.index, df_real_time['Power without DVFS'], label='Power without DVFS', color='blue', linestyle='--')
    axs[1].plot(df_real_time.index, df_real_time['Power with DVFS'], label='Power with DVFS', color='green')
    axs[1].plot(df_real_time.index, df_real_time['Power with DPM'], label='Power with DPM', color='orange')
    axs[1].plot(df_real_time.index, df_real_time['Power with AVS'], label='Power with AVS', color='red')
    axs[1].set_title('Real-Time Data: Power Consumption Comparison')
    axs[1].set_xlabel('Data Points')
    axs[1].set_ylabel('Power Consumption')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


plot_power_consumption(df, real_time_df)

def analyze_best_algorithm(performance_matrix):
    # Identify the best algorithm
    savings = performance_matrix[['Power Savings (DVFS)', 'Power Savings (DPM)', 'Power Savings (AVS)']]
    best_algorithm = savings.idxmax(axis=1).values[0]
    improvement_percent = savings.max(axis=1).values[0] / performance_matrix['Average Power without DVFS'].values[0] * 100

    return best_algorithm, improvement_percent

# Analyze best algorithm for historical data
best_algorithm_historical, improvement_percent_historical = analyze_best_algorithm(performance_matrix)

print("\nPerformance Matrix:\n", performance_matrix)
print(f"\nBest Algorithm (Historical Data): {best_algorithm_historical}")
print(f"Improvement Percentage (Historical Data): {improvement_percent_historical:.2f}%")

performance_matrix_real_time = calculate_performance_matrix(real_time_df)
best_algorithm_real_time, improvement_percent_real_time = analyze_best_algorithm(performance_matrix_real_time)

print("\nReal-Time Performance Matrix:\n", performance_matrix_real_time)
print(f"\nBest Algorithm (Real-Time Data): {best_algorithm_real_time}")
print(f"Improvement Percentage (Real-Time Data): {improvement_percent_real_time:.2f}%")
