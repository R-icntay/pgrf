# Create function for plotting the resliced image in an oblique view
import matplotlib.pyplot as plt
def plot_oblique_view(xx, yy, zz, slice_values, slice_idx, prostate_volume, save):
    """
    This is a function for plotting the resliced image in an oblique view.
    Parameters
    ----------
    xx : 2D array
        x-coordinates of the slice.
    yy : 2D array
        y-coordinates of the slice.
    zz : 2D array
        z-coordinates of the slice.
    slice_values : 2D array
        Intensity values of the slice.
    slice_idx : int
        Index of the slice to be extracted.
    prostate_volume : 3D array
        Input image.
    Returns
    -------
        A plot of the resliced image in an oblique view.
    """
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the x, y, and z limits of the plot
    x_len, y_len, z_len = prostate_volume.shape
    ax.set_xlim(0, x_len)
    ax.set_ylim(0, y_len)
    ax.set_zlim(0, z_len)

    # Create a 1D array of the slice values
    color_values = slice_values.flatten()

    # Create a 1D array of RGB colors corresponding to the intensity values
    cmap = plt.cm.gray
    norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    color_array = cmap(norm(color_values))

    # Plot the transformed slice as a 3D scatter plot
    ax.scatter(xx, yy, zz, c=color_array, s = 0.07)

    # Calculate the minimum and maximum z-coordinates of the transformed slice
    z_min = zz.min()
    z_max = zz.max()
    x_min = xx.min()
    x_max = xx.max()
    y_min = yy.min()
    y_max = yy.max()

    # Set the limits of the z-axis to include the entire slice
    ax.set_zlim(z_min, z_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set the title of the plot
    ax.set_title(f"{save} oblique view of slice {slice_idx} of a prostate volume")

    # Set the labels for the x, y, and z axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Save the plot
    plt.savefig(f"{save}_oblique_view_slice_{slice_idx}.png")

    # Close the plot
    plt.close()

    # Show the plot
    #plt.show()