import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def create_matrices(filename):
    df = pd.read_excel(filename)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces from column names
    df.columns = [col.strip() for col in df.columns]  # Ensure all columns are stripped

    num_of_nodes = len(df['Nodes'])
    num_of_stimuli = len(df['Stimuli'])
    stimuli_names = df['Stimuli'].tolist()
    node_names = df['Nodes'].tolist()
    
    mact = np.zeros((num_of_nodes, num_of_stimuli))
    minh = np.zeros((num_of_nodes, num_of_stimuli))
    
    for i in range(num_of_nodes):
        act_in_line = str(df['Activators'][i]).split(',')
        inh_in_line = str(df['Inhibitors'][i]).split(',')
        
        for j in range(num_of_stimuli):
            if stimuli_names[j] in act_in_line:
                mact[i, j] = 1
            if stimuli_names[j] in inh_in_line:
                minh[i, j] = 1
    
    return mact, minh, node_names, num_of_nodes, stimuli_names

def odesysfun(x, t, num_of_nodes, gamma, h, mact, minh, clamped):
    f = np.zeros(num_of_nodes)
    w = np.zeros(num_of_nodes)
    
    for i in range(num_of_nodes):
        ract = mact[i, :]
        rinh = minh[i, :]
        
        if not np.any(rinh) and np.any(ract):
            sum_alpha = np.sum(ract)
            sum_alpha_x = np.dot(ract, x)
            w[i] = ((1 + sum_alpha) / sum_alpha) * (sum_alpha_x / (1 + sum_alpha_x))
        
        elif not np.any(ract) and np.any(rinh):
            sum_beta = np.sum(rinh)
            sum_beta_x = np.dot(rinh, x)
            w[i] = 1 - ((1 + sum_beta) / sum_beta) * (sum_beta_x / (1 + sum_beta_x))
        
        elif np.any(ract) and np.any(rinh):
            sum_alpha = np.sum(ract)
            sum_beta = np.sum(rinh)
            sum_alpha_x = np.dot(ract, x)
            sum_beta_x = np.dot(rinh, x)
            w[i] = ((1 + sum_alpha) / sum_alpha) * (sum_alpha_x / (1 + sum_alpha_x)) * (1 - ((1 + sum_beta) / sum_beta) * (sum_beta_x / (1 + sum_beta_x)))
        
        else:
            w[i] = 0
        
        f[i] = (-np.exp(0.5 * h) + np.exp(-h * (w[i] - 0.5))) / ((1 - np.exp(0.5 * h)) * (1 + np.exp(-h * (w[i] - 0.5)))) - (gamma[i] * x[i])
        
        if clamped[i] == 1:
            f[i] = 0
    
    return f

# Network 1
mact, minh, node_names, num_of_nodes, stimuli_names = create_matrices('SMENR1.xlsx')
stimuli_names = ['TNF', 'IL-1β']
stimuli = [node_names.index(name) for name in stimuli_names]
stimuli.sort()

observed_names = ['ACAN', 'COL2A', 'COL10A1', 'COL1A', 'IFN-γ', 'TNF', 'IL-12A', 'IL-17A', 'IL-18', 'IL-1α', 'IL-1β', 'IL-6', 'IL-8', 'IL-1RA', 'IL-4', 'IL-10', 'TGF-β', 'IGF1', 'CSF2', 'GDF5', 'PGRN', 'CCL', 'CCL22', 'MMP1', 'MMP13', 'MMP2', 'MMP3', 'MMP9', 'VEGF', 'ADAMTS4/5', 'TIMP1/2', 'TIMP3']
responses = [node_names.index(name) for name in observed_names]
responses.sort()

t = np.linspace(0, 30, 100)
xinit = np.random.rand(num_of_nodes)
gamma = np.ones(num_of_nodes)
h = 10
clamped = np.zeros(num_of_nodes)

xout_final = np.zeros((100, num_of_nodes))
for i in range(100):
    xinit = np.random.rand(num_of_nodes)
    xout = odeint(odesysfun, xinit, t, args=(num_of_nodes, gamma, h, mact, minh, clamped))
    xout_final[i, :] = xout[-1, :]

xbaseline = np.mean(xout_final, axis=0)
median_value = np.median(xout_final, axis=0)
std_deviation = np.std(xout_final, axis=0)

xfinal = np.zeros((len(stimuli), num_of_nodes))
xdiff = np.zeros((len(stimuli), num_of_nodes))

for i, stim in enumerate(stimuli):
    xinit = np.random.rand(num_of_nodes)
    xinit[stim] = 1
    clamped[stim] = 1
    xout = odeint(odesysfun, xinit, t, args=(num_of_nodes, gamma, h, mact, minh, clamped))
    xfinal[i, :] = xout[-1, :]
    xdiff[i, :] = xfinal[i, :] - xbaseline

xdiff = xbaseline - xout[-1, :]
xout_mean = np.mean(xout, axis=0)

# Bar plot
colors = [
    [0.3010, 0.7450, 0.9330],
    [0.3010, 0.7450, 0.9330],
    [0.3010, 0.7450, 0.9330],
    [0.3010, 0.7450, 0.9330],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 1],
    [0.4660, 0.6740, 0.1880],
    [0.4660, 0.6740, 0.1880],
    [0.4660, 0.6740, 0.1880],
    [0.4660, 0.6740, 0.1880],
    [0.4660, 0.6740, 0.1880],
    [0.9290, 0.6940, 0.1250],
    [0.9290, 0.6940, 0.1250],
    [0.8500, 0.3250, 0.0980],
    [0.8500, 0.3250, 0.0980],
    [0.8500, 0.3250, 0.0980],
    [0.8500, 0.3250, 0.0980],
    [0.8500, 0.3250, 0.0980],
    [0.8500, 0.3250, 0.0980],
    [0.8500, 0.3250, 0.0980],
    [0.4940, 0.1840, 0.5560],
    [0.4940, 0.1840, 0.5560]
]

plt.figure(figsize=(20, 8))  # Adjust the figure size to make it wider
b = plt.bar(range(len(xbaseline)), xbaseline, color=colors)
plt.xticks(range(len(node_names)), node_names, rotation=90, fontsize=14, fontweight='bold')
plt.title('Median Baseline', fontsize=14)
plt.xlabel('Responses', fontsize=14)
plt.ylabel('Activation Level', fontsize=14)
plt.ylim([0, 1.05])
plt.savefig('median_baseline.png', bbox_inches='tight')  # Save plot to file

# Save table data to an Excel file
filename = 'Xbaseline.xlsx'
pd.DataFrame(xout_final).to_excel(filename, index=False)
print(f'Table data saved to {filename}')

# Model Stimulation - Inflammatory
stimuli_names = ['IL-1β']
stimuli = [node_names.index(name) for name in stimuli_names]
stimuli.sort()

t = np.linspace(0, 99, 100)
xinit = np.random.rand(num_of_nodes)
xinit[stimuli] = 1
clamped = np.zeros(num_of_nodes)
clamped[stimuli] = 1

xout_final = np.zeros((100, num_of_nodes))
for i in range(100):
    xout = odeint(odesysfun, xinit, t, args=(num_of_nodes, gamma, h, mact, minh, clamped))
    xout_final[i, :] = xout[-1, :]

xbaseline1 = np.mean(xout_final, axis=0)
xdiff = xbaseline1 - xout[-1, :]

# Plot Stimuli in bar plot
plt.figure(figsize=(20, 8))  # Adjust the figure size to make it wider
plt.bar(range(len(xbaseline)), xbaseline, 0.8, color=[0.3010, 0.7450, 0.9330])
plt.bar(range(len(xbaseline1)), xbaseline1, 0.3, color=[1, 1, 0])
plt.xticks(range(len(node_names)), node_names, rotation=90, fontsize=14, fontweight='bold')
plt.title('Stimuli IL-1β', fontsize=14)
plt.xlabel('Responses', fontsize=14)
plt.ylabel('Activation Level', fontsize=14)
plt.ylim([1e-9, 1.2])
plt.yscale('log')
plt.savefig('stimuli_IL-1β.png', bbox_inches='tight')  # Save plot to file

# Save table data to an Excel file
filename = 'Xbaseline1_IL-1β.xlsx'
pd.DataFrame(xout).to_excel(filename, index=False)
print(f'Table data saved to {filename}')
