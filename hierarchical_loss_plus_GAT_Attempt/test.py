import re
import csv

# Sample input data (you can read this from a file instead if needed)
data = """


"""

data = data.replace("Iteration", "\nIteration")

# Convert the results to float (optional)
returns = [float(r) for r in re.findall(r"Ep Return:\s*([0-9.+-eE]+)", data)]
indiv = [float(r) for r in re.findall(r"Action Return:\s*([0-9.+-eE]+)", data)]
len = [float(r) for r in re.findall(r"Ep Len:\s*([0-9.+-eE]+)", data)]
loss = [float(r) for r in re.findall(r"ALoss:\s*([0-9.+-eE]+)", data)]
vloss = [float(r) for r in re.findall(r"VLoss:\s*([0-9.+-eE]+)", data)]
cov = [float(r) for r in re.findall(r"cov:\s*([0-9.+-eE]+)", data)]
ent = [float(r) for r in re.findall(r"Entropy:\s*([0-9.+-eE]+)", data)]

# Output to CSV file
output_file = "average_returns.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Ep Return", "Sample Return", "Ep Len", "ALoss", "VLoss", "Cov", "Entropy"])  # Write header
    for i, el in enumerate(zip(returns, indiv, len, loss, vloss, cov, ent), 1):  # Start from iteration 84
        writer.writerow([i, *el])

print(f"Data saved to {output_file}")
