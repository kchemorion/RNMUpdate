import roadrunner
import matplotlib.pyplot as plt

# Load the SBML model
try:
    rr = roadrunner.RoadRunner('model.xml')
except Exception as e:
    print(f"Error loading SBML model: {e}")
    exit()

# Set simulation parameters
start_time = 0
end_time = 30
num_points = 100

# Run the simulation
try:
    results = rr.simulate(start_time, end_time, num_points)
except Exception as e:
    print(f"Error running simulation: {e}")
    exit()

# Print results structure and column names
print("Results structure:")
print(results)
print("Results columns:")
print(results.colnames)

# Extract time points
time_points = results[:, 0]

# Try to get the list of floating species IDs
try:
    species_ids = rr.model.getFloatingSpeciesIds()
    print("Floating species IDs:", species_ids)
except AttributeError:
    print("Error: Unable to get floating species IDs")
    exit()

# Ensure species_ids is not empty
if not species_ids:
    print("No floating species found in the model.")
    exit()

# Check if species names in results columns match species IDs
results_colnames = [name.strip('[]') for name in results.colnames[1:]]  # Remove '[]' from colnames
print("Processed results columns:", results_colnames)

# Check alignment of species IDs with results columns
species_alignment = [(species, species in results_colnames) for species in species_ids]
print("Species alignment with results columns:", species_alignment)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot each species concentration over time
for species in species_ids:
    try:
        if species in results_colnames:
            index = results.colnames.index(f'[{species}]')
            plt.plot(time_points, results[:, index], label=species)
        else:
            print(f"Species {species} not found in results columns")
    except Exception as e:
        print(f"Error plotting species {species}: {e}")

plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Simulation Results')
plt.legend()
plt.show()
