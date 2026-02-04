#!/usr/bin/env python3
import numpy as np

# Test histogram2d alignment
lon = np.array([-93, -93, -92, -92, -91])
lat = np.array([45, 45.5, 45, 45.5, 45])

h, xe, ye = np.histogram2d(lon, lat, bins=3)

with open('output/results/alignment_test.txt', 'w') as f:
    f.write(f"hist shape: {h.shape}\n")
    f.write(f"hist:\n{h}\n")
    f.write(f"xedges (lon): {xe}\n")
    f.write(f"yedges (lat): {ye}\n")

    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2
    f.write(f"x_centers: {xc}\n")
    f.write(f"y_centers: {yc}\n")

    X, Y = np.meshgrid(xc, yc)
    f.write(f"X shape: {X.shape}\n")
    f.write(f"Y shacpe: {Y.shape}\n")

    Z = h.T
    f.write(f"Z shape: {Z.shape}\n")
    f.write(f"Z:\n{Z}\n")

    f.write("\nKey insight: meshgrid returns (len(yc), len(xc)) arrays\n")
    f.write("hist2d returns (len(xc), len(yc)) array\n")
    f.write("So hist.T gives correct orientation for plot_surface\n")

print("Done - check output/results/alignment_test.txt")
