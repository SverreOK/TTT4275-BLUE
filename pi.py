import matplotlib.pyplot as plt
import numpy as np

# Define the number of sides for the polygons, starting from a triangle up to 100 sides
sides = np.arange(3, 25, 0.01)

# Calculate the inscribed perimeter for each n-sided polygon
inscribed_perimeters = 2 * sides * np.sin(np.pi / sides)

side_lengths = 2 * np.sin(np.pi / sides)

circle_sector_lengths = np.pi * np.ones(len(sides))

# Calculate the ratio of the side lenghts to the circle sector length
ratios = side_lengths / circle_sector_lengths


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sides, inscribed_perimeters, marker='o')
plt.xlabel('Number of Sides')
plt.ylabel('Inscribed Perimeter')
plt.title('Inscribed Perimeters of Polygons in the Unit Circle')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sides, side_lengths, marker='o')
plt.xlabel('Number of Sides')
plt.ylabel('Side Length')
plt.title('Side Lengths of Polygons in the Unit Circle')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sides, ratios, marker='o')
plt.xlabel('Number of Sides')
plt.ylabel('Side Length to Circle Sector Length Ratio')
plt.title('Ratio of Side Lengths to Circle Sector Lengths of Polygons in the Unit Circle')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.show() 