from chaco.api import PlotGrid, PlotAxis

def add_default_grids(plot, 
        major_index=None,
        minor_index=None,
        major_value=None, 
        minor_value=None):

    if major_index is not None:
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                orientation='vertical', line_style='solid',
                line_color='lightgray',
                grid_interval=major_index)
        plot.underlays.append(grid)

    if minor_index is not None:
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                orientation='vertical', line_style='dot',
                line_color='lightgray',
                grid_interval=minor_index)
        plot.underlays.append(grid)

    if major_value is not None:
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                orientation='horizontal', line_style='solid',
                line_color='lightgray',
                grid_interval=major_value)
        plot.underlays.append(grid)

    if minor_value is not None:
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                orientation='horizontal', line_style='dot',
                line_color='lightgray',
                grid_interval=minor_value)
        plot.underlays.append(grid)


def tick_formatter(s):
    return "{}:{:02}".format(*divmod(int(s), 60))


def add_time_axis(plot, orientation='bottom'):
    axis = PlotAxis(component=plot, 
                    orientation=orientation, 
                    title="Time (min:sec)",
                    tick_label_formatter=tick_formatter)
    plot.underlays.append(axis)
