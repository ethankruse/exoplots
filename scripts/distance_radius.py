import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import SelectionGeometry, Tap
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, FuncTickFormatter, TapTool
from bokeh.models import CustomJS, Label, LassoSelectTool, Legend, LegendItem
from bokeh.models import LogAxis, Range1d
from bokeh.models.widgets import Button
from bokeh.themes import Theme

from utils import csv_creation, get_update_time, log_axis_labels
from utils import deselect, openurl, palette, reset

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# what order to plot things and what the legend labels will say
missions = ['Kepler Candidate', 'Kepler Confirmed', 'K2 Candidate',
            'K2 Confirmed', 'TESS Candidate', 'Other Confirmed',
            'TESS Confirmed']

# markers and colors in the same order as the missions above
markers = ['circle_cross', 'circle', 'square_cross', 'square',
           'inverted_triangle', 'diamond', 'triangle']
# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
colors = [palette['C0'], palette['C0'], palette['C1'], palette['C1'],
          palette['C2'], palette['C3'], palette['C2']]

# output files
embedfile = '_includes/distance_radius_candidates_embed.html'
fullfile = '_includes/distance_radius_candidates.html'

# set up the full output file
plotting.output_file(fullfile, title='Distance Radius Plot')

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# what to display when hovering over a data point
TOOLTIPS = [
    ("Planet", "@planet"),
    # only give the decimal and sig figs if needed
    ("Distance", "@distance{0,0[.][0]} parsec"),
    ("Radius", "@radius{0,0[.][00]} Earth; @jupradius{0,0[.][0000]} Jup"),
    ("Period", "@period{0,0[.][0000]} days"),
    ("Discovered by", "@discovery"),
    ("Status", "@status")
]

# create the figure
fig = plotting.figure(x_axis_type='log', y_axis_type='log', tooltips=TOOLTIPS,
                      plot_height=700, plot_width=750)

# need to store min and max radius values to create the second axis
ymin = 1
ymax = 1
# save the output plots to rearrange them in the legend
glyphs = []
counts = []
sources = []
alphas = []
legitems = []

for ii, imiss in enumerate(missions):
    # select the appropriate set of planets for each mission
    # make the confirmed planets more opaque and bigger
    if imiss == 'Other Confirmed':
        good = ((~np.in1d(dfpl['facility'], ['Kepler', 'K2', 'TESS'])) &
                np.isfinite(dfpl['rade']) & np.isfinite(dfpl['distance_pc']) &
                dfpl['flag_tran'] & (dfpl['disposition'] == 'Confirmed'))
        alpha = 0.7
        size = 8
    else:
        fac = imiss.split()[0]
        disp = imiss.split()[1]
        good = ((dfpl['facility'] == fac) & np.isfinite(dfpl['rade']) &
                np.isfinite(dfpl['distance_pc']) & dfpl['flag_tran'] &
                (dfpl['disposition'] == disp))
        if disp == 'Confirmed':
            alpha = 0.7
            size = 6
            if 'TESS' == fac:
                size = 8
        else:
            # candidates get these values
            alpha = 0.35
            size = 4
            # make the yellow darker
            if fac == 'TESS':
                alpha = 0.6
                size = 5

    counts.append(f'{good.sum():,}')

    # what the hover tooltip draws its values from
    source = plotting.ColumnDataSource(data=dict(
            planet=dfpl['name'][good],
            distance=dfpl['distance_pc'][good],
            radius=dfpl['rade'][good],
            jupradius=dfpl['radj'][good],
            period=dfpl['period'][good],
            host=dfpl['hostname'][good],
            discovery=dfpl['facility'][good],
            status=dfpl['disposition'][good],
            url=dfpl['url'][good]
            ))
    print(imiss, ': ', good.sum())

    # plot the planets
    # nonselection stuff is needed to prevent planets in that category from
    # disappearing when you click on a data point ("select" it)
    glyph = fig.scatter('distance', 'radius', color=colors[ii], source=source,
                        size=size, alpha=alpha, marker=markers[ii],
                        nonselection_alpha=0.07, selection_alpha=0.8,
                        nonselection_color=colors[ii])
    glyphs.append(glyph)
    sources.append(source)
    alphas.append(alpha)
    # save the global min/max
    ymin = min(ymin, source.data['radius'].min())
    ymax = max(ymax, source.data['radius'].max())
    leg = LegendItem(label=imiss + f' ({counts[ii]})', renderers=[glyph])
    legitems.append(leg)

# figure out what the default axis limits are
ydiff = np.log10(ymax) - np.log10(ymin)
ystart = 10.**(np.log10(ymin) - 0.05*ydiff)
yend = 10.**(np.log10(ymax) + 0.05*ydiff)

# jupiter/earth radius ratio
radratio = 11.21

# set up the second axis with the proper scaling
fig.extra_y_ranges = {"jup": Range1d(start=ystart/radratio, end=yend/radratio)}
fig.add_layout(LogAxis(y_range_name="jup"), 'right')

# add the first y-axis's label and use our custom log formatting for both axes
fig.yaxis.axis_label = 'Radius (Earth Radii)'
fig.yaxis.formatter = FuncTickFormatter(code=log_axis_labels())

# add the x-axis's label and use our custom log formatting
fig.xaxis.axis_label = 'Distance (parsec)'
fig.xaxis.formatter = FuncTickFormatter(code=log_axis_labels())

# add the second y-axis's label
fig.right[0].axis_label = 'Radius (Jupiter Radii)'

# which order to place the legend labels
topleg = ['Kepler Confirmed', 'K2 Confirmed', 'TESS Confirmed']
bottomleg = ['Kepler Candidate', 'K2 Candidate', 'TESS Candidate']
vbottomleg = ['Other Confirmed']

# set up all the legend objects
items1 = [legitems[missions.index(ii)] for ii in topleg]
items2 = [legitems[missions.index(ii)] for ii in bottomleg]
items3 = [legitems[missions.index(ii)] for ii in vbottomleg]

# create the two legends
for ii in np.arange(3):
    if ii == 0:
        items = items3
    elif ii == 1:
        items = items2
    else:
        items = items1
    legend = Legend(items=items, location="center")

    if ii == 2:
        legend.title = 'Discovered by and Status'
        legend.spacing = 10
    else:
        legend.spacing = 11

    legend.location = (-70, 5)
    legend.label_text_align = 'left'
    legend.margin = 0

    fig.add_layout(legend, 'above')

# overall figure title
fig.title.text = 'Transiting Planets and Planet Candidates'

# create the four lines of credit text in the two bottom corners
label_opts1 = dict(
    x=-85, y=32,
    x_units='screen', y_units='screen'
)

label_opts2 = dict(
    x=-85, y=37,
    x_units='screen', y_units='screen'
)

label_opts3 = dict(
    x=612, y=69,
    x_units='screen', y_units='screen', text_align='right',
    text_font_size='9pt'
)

label_opts4 = dict(
    x=612, y=73,
    x_units='screen', y_units='screen', text_align='right',
    text_font_size='9pt'
)

msg1 = 'By Exoplots'
# when did the data last get updated
modtimestr = get_update_time().strftime('%Y %b %d')
msg3 = 'Data: NASA Exoplanet Archive'
msg4 = 'and ExoFOP-TESS'

caption1 = Label(text=msg1, **label_opts1)
caption2 = Label(text=modtimestr, **label_opts2)
caption3 = Label(text=msg3, **label_opts3)
caption4 = Label(text=msg4, **label_opts4)

fig.add_layout(caption1, 'below')
fig.add_layout(caption2, 'below')
fig.add_layout(caption3, 'below')
fig.add_layout(caption4, 'below')

# add the download button
button = Button(label="Download CSV of Selected Data", button_type="primary")
# what is the header and what keys correspond to those columns for
# output CSV files
keys = source.column_names
keys.remove('url')
keys.remove('host')
csvhead = '# ' + ', '.join(keys)
button.js_on_click(CustomJS(args=dict(sources=sources, keys=keys,
                                      header=csvhead), code=csv_creation))

# select multiple points to download
box = BoxSelectTool()
fig.add_tools(box)
lasso = LassoSelectTool()
fig.add_tools(lasso)

des = CustomJS(args=dict(glyphs=glyphs, alphas=alphas, legends=legitems),
               code=deselect)
fig.js_on_event(SelectionGeometry, des)
fig.js_on_event('reset', CustomJS(args=dict(glyphs=glyphs, alphas=alphas,
                                            legends=legitems), code=reset))
for iglyph in glyphs:
    iglyph.js_on_change('visible', des)

# allow for something to happen when you click on data points
fig.add_tools(TapTool())
# set up where to send people when they click on a planet
uclick = CustomJS(args=dict(sources=sources), code=openurl)
fig.js_on_event(Tap, uclick)

layout = column(button, fig)

plotting.save(layout)

plotting.show(layout)

# save the individual pieces so we can just embed the figure without the whole
# html page
script, div = components(layout, theme=theme)
with open(embedfile, 'w') as ff:
    ff.write(script)
    ff.write(div)
