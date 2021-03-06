"""
This file patches seaborn's letter value function so that outliers are not
plotted. Unfortunately, seaborn does not provide an option to disable this
otherwise.
"""
import matplotlib.patches as Patches
import matplotlib as mpl
import numpy as np
import seaborn.categorical


def draw_letter_value_plot(self, ax, kws):
    """Use matplotlib to draw a letter value plot on an Axes."""
    vert = self.orient == "v"

    for i, group_data in enumerate(self.plot_data):

        if self.plot_hues is None:

            # Handle case where there is data at this level
            if group_data.size == 0:
                continue

            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_na(group_data)

            # Handle case where there is no non-null data
            if box_data.size == 0:
                continue

            color = self.colors[i]

            self._lvplot(box_data,
                         positions=[i],
                         color=color,
                         vert=vert,
                         widths=self.width,
                         k_depth=self.k_depth,
                         ax=ax,
                         scale=self.scale,
                         outlier_prop=self.outlier_prop,
                         **kws)

        else:
            # Draw nested groups of boxes
            offsets = self.hue_offsets
            for j, hue_level in enumerate(self.hue_names):

                # Add a legend for this hue level
                if not i:
                    self.add_legend_data(ax, self.colors[j], hue_level)

                # Handle case where there is data at this level
                if group_data.size == 0:
                    continue

                hue_mask = self.plot_hues[i] == hue_level
                box_data = remove_na(group_data[hue_mask])

                # Handle case where there is no non-null data
                if box_data.size == 0:
                    continue

                color = self.colors[j]
                center = i * k + offsets[j]
                self._lvplot(box_data,
                             positions=[center],
                             color=color,
                             vert=vert,
                             widths=self.nested_width,
                             k_depth=self.k_depth,
                             ax=ax,
                             scale=self.scale,
                             outlier_prop=self.outlier_prop,
                             **kws)
seaborn.categorical.draw_letter_value_plot = draw_letter_value_plot

def _lvplot(self, box_data, positions,
            color=[255. / 256., 185. / 256., 0.],
            vert=True, widths=1, k_depth='proportion',
            ax=None, outlier_prop=None, scale='exponential',
            **kws):

    x = positions[0]
    box_data = np.asarray(box_data)

    # If we only have one data point, plot a line
    if len(box_data) == 1:
        kws.update({'color': self.gray, 'linestyle': '-'})
        ys = [box_data[0], box_data[0]]
        xs = [x - widths / 2, x + widths / 2]
        if vert:
            xx, yy = xs, ys
        else:
            xx, yy = ys, xs
        ax.plot(xx, yy, **kws)
    else:
        # Get the number of data points and calculate "depth" of
        # letter-value plot
        box_ends, k = self._lv_box_ends(box_data, k_depth=k_depth,
                                        outlier_prop=outlier_prop)

        # Anonymous functions for calculating the width and height
        # of the letter value boxes
        width = self._width_functions(scale)

        # Function to find height of boxes
        def height(b):
            return b[1] - b[0]

        # Functions to construct the letter value boxes
        def vert_perc_box(x, b, i, k, w):
            rect = Patches.Rectangle((x - widths*w / 2, b[0]),
                                     widths*w,
                                     height(b), fill=True)
            return rect

        def horz_perc_box(x, b, i, k, w):
            rect = Patches.Rectangle((b[0], x - widths*w / 2),
                                     height(b), widths*w,
                                     fill=True)
            return rect

        # Scale the width of the boxes so the biggest starts at 1
        w_area = np.array([width(height(b), i, k)
                           for i, b in enumerate(box_ends)])
        w_area = w_area / np.max(w_area)

        # Calculate the medians
        y = np.median(box_data)

        # Calculate the outliers and plot
        outliers = self._lv_outliers(box_data, k)

        if vert:
            boxes = [vert_perc_box(x, b[0], i, k, b[1])
                     for i, b in enumerate(zip(box_ends, w_area))]

            # Plot the medians
            ax.plot([x - widths / 2, x + widths / 2], [y, y],
                    c='.15', alpha=.45, **kws)

#                ax.scatter(np.repeat(x, len(outliers)), outliers,
#                           marker='d', c=mpl.colors.rgb2hex(color), **kws)
        else:
            boxes = [horz_perc_box(x, b[0], i, k, b[1])
                     for i, b in enumerate(zip(box_ends, w_area))]

            # Plot the medians
            ax.plot([y, y], [x - widths / 2, x + widths / 2],
                    c='.15', alpha=.45, **kws)

            ax.scatter(outliers, np.repeat(x, len(outliers)),
                       marker='d', c=color, **kws)

        # Construct a color map from the input color
        rgb = [[1, 1, 1], list(color)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('new_map', rgb)
        collection = mpl.collections.PatchCollection(boxes, cmap=cmap)

        # Set the color gradation
        collection.set_array(np.array(np.linspace(0, 1, len(boxes))))

        # Plot the boxes
        ax.add_collection(collection)

seaborn.categorical._LVPlotter._lvplot = _lvplot
