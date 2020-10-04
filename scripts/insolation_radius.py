import numpy as np
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import SelectionGeometry
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, FuncTickFormatter, OpenURL, TapTool
from bokeh.models import CustomJS, Label, LassoSelectTool, Legend, LegendItem
from bokeh.models import LogAxis, Range1d
from bokeh.models.widgets import Button
from bokeh.themes import Theme

from utils import csv_creation, get_update_time, load_data, log_axis_labels
from utils import deselect, reset

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
colors = ['#228833', '#228833', '#ee6677', '#ee6677', '#ccbb44', '#aa3377',
          '#ccbb44']

# load the data
new = True
dfcon, dfkoi, dfk2, dftoi, comp = load_data(updated_koi_params=True,
                                      updated_k2_params=True, new=new)

if new:
    fackey = 'disc_facility'
    trankey = 'tran_flag'
    hostkey = 'hostname'
else:
    fackey = 'pl_facility'
    trankey = 'pl_tranflag'
    hostkey = 'pl_hostname'

# what to display when hovering over a data point
TOOLTIPS = [
    ("Planet", "@planet"),
    # only give the decimal and sig figs if needed
    ("Insolation", "@insolation{0,0[.][00]} Earths"),
    ("Radius", "@radius{0,0[.][00]} Earth; @jupradius{0,0[.][0000]} Jup"),
    ("Period", "@period{0,0[.][0000]} days"),
    ("Discovered by", "@discovery"),
    ("Status", "@status")
]

for ifig in np.arange(2):

    # output files
    if ifig == 0:
        embedfile = '_includes/insolation_radius_candidates_embed.html'
        fullfile = '_includes/insolation_radius_candidates.html'
    else:
        embedfile = '_includes/radius_gap_embed.html'
        fullfile = '_includes/radius_gap.html'

    # set up the full output file
    plotting.output_file(fullfile, title='Insolation Radius Plot')

    # create the figure
    if ifig == 0:
        fig = plotting.figure(x_axis_type='log', y_axis_type='log',
                              tooltips=TOOLTIPS, plot_height=700)
    else:
        fig = plotting.figure(x_axis_type='log', y_axis_type='linear',
                              tooltips=TOOLTIPS, plot_height=700)
    # allow for something to happen when you click on data points
    fig.add_tools(TapTool())

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
        # candidates get these default values
        alpha = 0.35
        size = 4
        # select the appropriate set of planets for each mission
        # make the confirmed planets more opaque and bigger
        if imiss == 'Other Confirmed':
            good = ((~np.in1d(dfcon[fackey], ['Kepler', 'K2', 'TESS'])) &
                    np.isfinite(dfcon['pl_rade']) &
                    np.isfinite(dfcon['insol']) &
                    dfcon[trankey].astype(bool))
            alpha = 0.7
            size = 8
        elif 'Confirmed' in imiss:
            fac = imiss.split()[0]
            good = ((dfcon[fackey] == fac) &
                    np.isfinite(dfcon['pl_rade']) &
                    np.isfinite(dfcon['insol']) &
                    dfcon[trankey].astype(bool))
            alpha = 0.7
            size = 6
        elif 'Kepler' in imiss:
            good = ((dfkoi['koi_disposition'] == 'Candidate') &
                    np.isfinite(dfkoi['insol']) &
                    np.isfinite(dfkoi['koi_prad']))
            # what the hover tooltip draws its values from
            source = plotting.ColumnDataSource(data=dict(
                    planet=dfkoi['kepoi_name'][good],
                    insolation=dfkoi['insol'][good],
                    period=dfkoi['koi_period'][good],
                    radius=dfkoi['koi_prad'][good],
                    jupradius=dfkoi['koi_pradj'][good],
                    host=dfkoi['kepid'][good],
                    discovery=dfkoi['pl_facility'][good],
                    status=dfkoi['koi_disposition'][good],
                    url=dfkoi['url'][good]
                    ))
            print(imiss, ': ', good.sum())
        elif 'K2' in imiss:
            good = ((dfk2['k2c_disp'] == 'Candidate') &
                    np.isfinite(dfk2['pl_rade']) &
                    np.isfinite(dfk2['insol']) &
                    dfk2['k2c_recentflag'].astype(bool))
            # what the hover tooltip draws its values from
            source = plotting.ColumnDataSource(data=dict(
                    planet=dfk2['epic_candname'][good],
                    insolation=dfk2['insol'][good],
                    period=dfk2['pl_orbper'][good],
                    radius=dfk2['pl_rade'][good],
                    jupradius=dfk2['pl_radj'][good],
                    host=dfk2['epic_name'][good],
                    discovery=dfk2['pl_facility'][good],
                    status=dfk2['k2c_disp'][good],
                    url=dfk2['url'][good]
                    ))
            print(imiss, ': ', good.sum())
        else:
            good = ((dftoi['disp'] == 'Candidate') &
                    np.isfinite(dftoi['prade']) &
                    np.isfinite(dftoi['insol']))
            # what the hover tooltip draws its values from
            source = plotting.ColumnDataSource(data=dict(
                    planet=dftoi['TOI'][good],
                    insolation=dftoi['insol'][good],
                    period=dftoi['period'][good],
                    radius=dftoi['prade'][good],
                    jupradius=dftoi['pradj'][good],
                    host=dftoi['host'][good],
                    discovery=dftoi['pl_facility'][good],
                    status=dftoi['disp'][good],
                    url=dftoi['url'][good]
                    ))
            print(imiss, ': ', good.sum())
            alpha = 0.6
        counts.append(f'{good.sum():,}')

        if 'Confirmed' in imiss:
            # what the hover tooltip draws its values from
            source = plotting.ColumnDataSource(data=dict(
                    planet=dfcon['pl_name'][good],
                    insolation=dfcon['insol'][good],
                    period=dfcon['pl_orbper'][good],
                    radius=dfcon['pl_rade'][good],
                    jupradius=dfcon['pl_radj'][good],
                    host=dfcon[hostkey][good],
                    discovery=dfcon[fackey][good],
                    status=dfcon['status'][good],
                    url=dfcon['url'][good]
                    ))
            print(imiss, ': ', good.sum())

        # plot the planets
        # nonselection stuff is needed to prevent planets in that category from
        # disappearing when you click on a data point ("select" it)
        glyph = fig.scatter('insolation', 'radius', color=colors[ii],
                            source=source, size=size, alpha=alpha,
                            marker=markers[ii], nonselection_alpha=0.07,
                            selection_alpha=0.8, nonselection_color=colors[ii])
        glyphs.append(glyph)
        sources.append(source)
        alphas.append(alpha)
        # save the global min/max
        ymin = min(ymin, source.data['radius'].min())
        ymax = max(ymax, source.data['radius'].max())

        leg = LegendItem(label=imiss + f' ({counts[ii]})', renderers=[glyph])
        legitems.append(leg)

    # set up where to send people when they click on a planet
    url = "@url"
    taptool = fig.select(TapTool)
    taptool.callback = OpenURL(url=url)

    # figure out what the default axis limits are
    if ifig == 0:
        ydiff = np.log10(ymax) - np.log10(ymin)
        ystart = 10.**(np.log10(ymin) - 0.05*ydiff)
        yend = 10.**(np.log10(ymax) + 0.05*ydiff)
    else:
        ystart = 0
        yend = 3.6

    # jupiter/earth radius ratio
    radratio = 11.21

    # set up the second axis with the proper scaling
    if ifig == 0:
        fig.extra_y_ranges = {"jup": Range1d(start=ystart/radratio,
                                             end=yend/radratio)}
        fig.add_layout(LogAxis(y_range_name="jup"), 'right')
    else:
        fig.y_range.start = ystart
        fig.y_range.end = yend
        fig.x_range.start = 40000
        fig.x_range.end = 0.1

    # add the first y-axis's label and use our custom log formatting
    fig.yaxis.axis_label = 'Radius (Earth Radii)'
    if ifig == 0:
        fig.yaxis.formatter = FuncTickFormatter(code=log_axis_labels())

    # add the x-axis's label and use our custom log formatting
    fig.xaxis.axis_label = 'Insolation (Earths)'
    fig.xaxis.formatter = FuncTickFormatter(code=log_axis_labels())
    # make high insolations on the left
    fig.x_range.flipped = True

    # add the second y-axis's label
    if ifig == 0:
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

        if ifig == 0:
            legend.location = (-70, 5)
        else:
            legend.location = (-50, 5)
        legend.label_text_align = 'left'
        legend.margin = 0

        fig.add_layout(legend, 'above')

    # overall figure title
    fig.title.text = 'Transiting Planets and Planet Candidates'

    # create the four lines of credit text in the two bottom corners
    if ifig == 0:
        label_opts1 = dict(
            x=-85, y=42,
            x_units='screen', y_units='screen'
        )

        label_opts2 = dict(
            x=-85, y=47,
            x_units='screen', y_units='screen'
        )

        label_opts3 = dict(
            x=612, y=79,
            x_units='screen', y_units='screen', text_align='right',
            text_font_size='9pt'
        )

        label_opts4 = dict(
            x=612, y=83,
            x_units='screen', y_units='screen', text_align='right',
            text_font_size='9pt'
        )
    else:
        label_opts1 = dict(
            x=-65, y=42,
            x_units='screen', y_units='screen'
        )

        label_opts2 = dict(
            x=-65, y=47,
            x_units='screen', y_units='screen'
        )

        label_opts3 = dict(
            x=632, y=79,
            x_units='screen', y_units='screen', text_align='right',
            text_font_size='9pt'
        )

        label_opts4 = dict(
            x=632, y=83,
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
    button = Button(label="Download CSV of Selected Data",
                    button_type="primary")
    # what is the header and what keys correspond to those columns for
    # output CSV files
    csvhead = '# Planet Name, Status, Period (days), Radius (Earths), ' \
              'Radius (Jupiters), Insolation (Earths), Discovered by'
    keys = ['planet', 'status', 'period', 'radius', 'jupradius', 'insolation',
            'discovery']
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

    layout = column(button, fig)

    plotting.save(layout)

    plotting.show(layout)

    # save the individual pieces so we can just embed the figure without the
    # whole html page
    script, div = components(layout, theme=theme)
    with open(embedfile, 'w') as ff:
        ff.write(script)
        ff.write(div)
