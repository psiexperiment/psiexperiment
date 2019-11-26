# This script creates the icon image. To trim down afterwards:
#   convert psi-logo.png -trim -fuzz 1 psi-logo-trimmed.png
import matplotlib as mp
import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 1)
lw = 10
adj = 2.5
r = 2

patch = mp.patches.Rectangle((-10, -20), 30, 30, facecolor='white')
axes.add_patch(patch)

patch = mp.patches.Circle((0, 0), radius=2, lw=lw, edgecolor='black', facecolor='none')
axes.add_patch(patch)

patch = mp.patches.Circle((5, 5), radius=2, lw=lw, edgecolor='black', facecolor='none')
axes.add_patch(patch)

patch = mp.patches.Circle((10, 0), radius=2, lw=lw, edgecolor='black', facecolor='none')
axes.add_patch(patch)

patch = mp.patches.Circle((5, -15), radius=2, lw=lw, edgecolor='black', facecolor='none')
axes.add_patch(patch)

axes.plot([5, 5], [-15+adj, 5-adj], 'k-', lw=lw)
axes.plot([0, 0, 2.5, 5, 7.5, 10, 10], [0-adj, -5, -7.5, -7.5, -7.5, -5, 0-adj], 'k-', lw=lw)

# If we want a circlej
#patch = mp.patches.Circle((5, -5), radius=15, lw=lw, edgecolor='black', facecolor='none')
#axes.add_patch(patch)

# If we want a square
patch = mp.patches.Circle((-10, 10), radius=2, lw=lw, edgecolor='black', facecolor='none')
axes.add_patch(patch)
patch = mp.patches.Circle((20, -20), radius=2, lw=lw, edgecolor='black', facecolor='none')
axes.add_patch(patch)
axes.plot([-10+adj, 20, 20], [10, 10, -20+adj], 'k-', lw=lw)
axes.plot([-10, -10, 20-adj], [10-adj, -20, -20], 'k-', lw=lw)

axes.axis('equal')
axes.set_axis_off()

figure.savefig('psi-logo.png', transparent=True, bbox_inches='tight')
