from datetime import datetime

import numpy as np
import pandas as pd
from astropy import constants as const
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import DoubleTap, SelectionGeometry, Tap
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BoxSelectTool, Button, ColorPicker, CustomJS, Div
from bokeh.models import ImageURL, Label, LassoSelectTool, Legend, LegendItem
from bokeh.models import LogAxis, Range1d, RangeSlider, TapTool, Toggle
from bokeh.themes import Theme

from utils import change_color, csv_creation, deselect, get_update_time
from utils import openurl, palette, playpause, reset, sliderselect, SolarSystem
from utils import unselect, yearselect

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# in what order to plot things and what the legend labels will say
missions = ['Kepler Candidate', 'K2 Candidate', 'TESS Candidate',
            'Radial Velocity', 'Other Transit', 'Kepler Confirmed',
            'K2 Confirmed', 'TESS Confirmed', 'Other Methods']

# markers and colors in the same order as the missions above
markers = ['circle_cross', 'square_cross', 'inverted_triangle', 'plus',
           'square_pin', 'circle', 'square', 'triangle', 'diamond']
# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
colors = [palette['C0'], palette['C1'], palette['C2'], palette['C4'],
          palette['C5'], palette['C0'], palette['C1'], palette['C2'],
          palette['C3']]

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

for ifig in np.arange(4):
    if ifig == 0:
        # output files
        embedfile = '_includes/period_radius_all_candidates_embed.html'
        fullfile = '_includes/period_radius_all_candidates.html'
    elif ifig == 1:
        embedfile = '_includes/period_radius_all_candidates_year_embed.html'
        fullfile = '_includes/period_radius_all_candidates_year.html'
    elif ifig == 2:
        # output files
        embedfile = '_includes/period_mass_all_candidates_embed.html'
        fullfile = '_includes/period_mass_all_candidates.html'
    else:
        embedfile = '_includes/period_mass_all_candidates_year_embed.html'
        fullfile = '_includes/period_mass_all_candidates_year.html'

    # what to display when hovering over a data point
    TOOLTIPS = [
        ("Planet", "@planet"),
        # only give the decimal and sig figs if needed
        ("Period", "@period{0,0[.][0000]} days"),
        ("Measured Radius",
         "@radius{0,0[.][00]} Earth; @jupradius{0,0[.][0000]} Jup"),
        ("Estimated Radius",
         "@radius_est{0,0[.][00]} Earth; @jupradius_est{0,0[.][0000]} Jup"),
        ("Measured Mass",
         "@mass{0,0[.][00]} Earth; @jupmass{0,0[.][0000]} Jup"),
        ("Estimated Mass",
         "@mass_est{0,0[.][00]} Earth; @jupmass_est{0,0[.][0000]} Jup"),
        ("Discovered via", "@method"),
        ("Discovered in", "@discyear"),
        ("Confirmed in", "@confyear")
    ]

    # set up the full output file
    plotting.output_file(fullfile, title='Period Radius Mass Plot')

    # create the figure
    fig = plotting.figure(x_axis_type='log', y_axis_type='log',
                          height=700, width=750, tooltips=TOOLTIPS)

    # need to store min and max radius values to create the second axis
    ymin = 1
    ymax = 1
    xmin = 1
    xmax = 1
    # save the output plots to rearrange them in the legend
    glyphs = []
    counts = []
    sources = []
    alphas = []
    legitems = []
    minyr = 2020

    solarsys = SolarSystem()

    solarsys.data['width'] = 15 * solarsys.width_mults
    solarsys.data['height'] = 15 * solarsys.width_mults
    source = plotting.ColumnDataSource(data=solarsys.data)
    if ifig < 2:
        image1 = ImageURL(url="url", x="period", y="radius", w="width",
                          h="height", anchor="center", w_units="screen",
                          h_units="screen")
    else:
        image1 = ImageURL(url="url", x="period", y="mass", w="width",
                          h="height", anchor="center", w_units="screen",
                          h_units="screen")
    gg = fig.add_glyph(source, image1)
    solarleg = LegendItem(label='Solar System', renderers=[gg])

    for ii, imiss in enumerate(missions):
        # select the appropriate set of planets for each mission
        # make the confirmed planets more opaque and bigger
        if imiss == 'Other Transit':
            good = ((~np.in1d(dfpl['facility'], ['Kepler', 'K2', 'TESS'])) &
                    (np.isfinite(dfpl['rade']) |
                     np.isfinite(dfpl['rade_est'])) &
                    (np.isfinite(dfpl['masse']) |
                     np.isfinite(dfpl['masse_est'])) &
                    np.isfinite(dfpl['period']) &
                    (dfpl['disposition'] == 'Confirmed') &
                    (dfpl['discoverymethod'] == 'Transit'))
            alpha = 0.7
            size = 8
        elif imiss == 'Radial Velocity':
            good = ((~np.in1d(dfpl['facility'], ['Kepler', 'K2', 'TESS'])) &
                    (np.isfinite(dfpl['rade']) |
                     np.isfinite(dfpl['rade_est'])) &
                    (np.isfinite(dfpl['masse']) |
                     np.isfinite(dfpl['masse_est'])) &
                    np.isfinite(dfpl['period']) &
                    (dfpl['disposition'] == 'Confirmed') &
                    (dfpl['discoverymethod'] == 'Radial Velocity'))
            alpha = 0.7
            size = 8
        elif imiss == 'Other Methods':
            good = ((~np.in1d(dfpl['facility'], ['Kepler', 'K2', 'TESS'])) &
                    (np.isfinite(dfpl['rade']) |
                     np.isfinite(dfpl['rade_est'])) &
                    (np.isfinite(dfpl['masse']) |
                     np.isfinite(dfpl['masse_est'])) &
                    np.isfinite(dfpl['period']) &
                    (dfpl['disposition'] == 'Confirmed') &
                    (~np.in1d(dfpl['discoverymethod'],
                              ['Radial Velocity', 'Transit'])))
            alpha = 0.7
            size = 8
        else:
            fac = imiss.split()[0]
            disp = imiss.split()[1]
            good = ((dfpl['facility'] == fac) &
                    (np.isfinite(dfpl['rade']) |
                     np.isfinite(dfpl['rade_est'])) &
                    (np.isfinite(dfpl['masse']) |
                     np.isfinite(dfpl['masse_est'])) &
                    np.isfinite(dfpl['period']) & (dfpl['disposition'] == disp))
            if disp == 'Confirmed':
                alpha = 0.7
                size = 6
            else:
                # candidates get these values
                alpha = 0.35
                size = 4
                # make the yellow darker
                if fac == 'TESS':
                    alpha = 0.6

        counts.append(f'{good.sum():,}')

        # what the hover tooltip draws its values from
        source = plotting.ColumnDataSource(data=dict(
                planet=dfpl['name'][good],
                period=dfpl['period'][good],
                radius_plot=np.nanmin(dfpl.loc[good, ['rade', 'rade_est']],
                                      axis=1),
                radius=dfpl['rade'][good],
                radius_est=dfpl['rade_est'][good],
                jupradius=dfpl['radj'][good],
                jupradius_est=dfpl['radj_est'][good],
                mass_plot=np.nanmin(dfpl.loc[good, ['masse', 'masse_est']],
                                    axis=1),
                mass=dfpl['masse'][good],
                mass_est=dfpl['masse_est'][good],
                jupmass=dfpl['massj'][good],
                jupmass_est=dfpl['massj_est'][good],
                host=dfpl['hostname'][good],
                method=dfpl['discoverymethod'][good],
                discovery=dfpl['facility'][good],
                url=dfpl['url'][good],
                year=dfpl['year_discovered'][good],
                discyear=dfpl['year_discovered'][good],
                confyear=dfpl['year_confirmed'][good]
                ))

        print(imiss, ': ', good.sum())

        # plot the planets
        # nonselection stuff is needed to prevent planets in that category from
        # disappearing when you click on a data point ("select" it)
        # selection_alpha is just requiring it to make its own selection glyph,
        # so we can better control that later
        if ifig == 0:
            glyph = fig.scatter('period', 'radius_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.07,
                                selection_alpha=0.8,
                                nonselection_color=colors[ii])
        elif ifig == 1:
            glyph = fig.scatter('period', 'radius_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.,
                                selection_alpha=alpha,
                                nonselection_color=colors[ii])
        elif ifig == 2:
            glyph = fig.scatter('period', 'mass_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.07,
                                selection_alpha=0.8,
                                nonselection_color=colors[ii])
        else:
            glyph = fig.scatter('period', 'mass_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.,
                                selection_alpha=alpha,
                                nonselection_color=colors[ii])
        glyphs.append(glyph)
        sources.append(source)
        alphas.append(alpha)
        # save the global min/max
        if ifig < 2:
            ymin = min(ymin, source.data['radius_plot'].min())
            ymax = max(ymax, source.data['radius_plot'].max())
        else:
            ymin = min(ymin, source.data['mass_plot'].min())
            ymax = max(ymax, source.data['mass_plot'].max())
        ab0 = source.data['period'][source.data['period'] > 0]
        xmin = min(xmin, ab0.min())
        xmax = max(xmax, source.data['period'].max())

        leg = LegendItem(label=imiss + f' ({counts[ii]})', renderers=[glyph])
        legitems.append(leg)
        minyr = int(min(minyr, source.data['discyear'].min()))

    # figure out what the default axis limits are
    if ifig < 2:
        ymax = min(ymax, 10. * (const.R_jup / const.R_earth).value)
    else:
        ymax = min(ymax, 0.08 * (const.M_sun / const.M_earth).value)
    ydiff = np.log10(ymax) - np.log10(ymin)
    ystart = 10.**(np.log10(ymin) - 0.05*ydiff)
    yend = 10.**(np.log10(ymax) + 0.05*ydiff)
    fig.y_range = Range1d(start=ystart, end=yend)

    xdiff = np.log10(xmax) - np.log10(xmin)
    xstart = 10.**(np.log10(xmin) - 0.05*xdiff)
    xend = 10.**(np.log10(xmax) + 0.05*xdiff)

    fig.x_range = Range1d(start=xstart, end=xend)

    # jupiter/earth radius ratio
    radratio = (const.R_jup / const.R_earth).value
    massratio = (const.M_jup / const.M_earth).value

    # set up the second axis with the proper scaling
    if ifig < 2:
        fig.extra_y_ranges = {"jup": Range1d(start=ystart/radratio,
                                             end=yend/radratio)}
    else:
        fig.extra_y_ranges = {"jup": Range1d(start=ystart/massratio,
                                             end=yend/massratio)}
    fig.add_layout(LogAxis(y_range_name="jup"), 'right')

    # add the first y-axis's label
    if ifig < 2:
        fig.yaxis.axis_label = r'\[\text{Radius} (\mathrm{R_\oplus})\]'
    else:
        fig.yaxis.axis_label = r'\[\text{Mass} (\mathrm{M_\oplus})\]'
    fig.yaxis.formatter.min_exponent = 3

    # add the x-axis's label
    fig.xaxis.axis_label = 'Period (days)'
    fig.xaxis.formatter.min_exponent = 3

    # add the second y-axis's label
    if ifig < 2:
        fig.right[0].axis_label = r'\[\text{Radius} (\mathrm{R_J})\]'
    else:
        fig.right[0].axis_label = r'\[\text{Mass} (\mathrm{M_J})\]'

    # which order to place the legend labels
    topleg = ['Kepler Confirmed', 'K2 Confirmed', 'TESS Confirmed']
    bottomleg = ['Kepler Candidate', 'K2 Candidate', 'TESS Candidate']
    vbottomleg = ['Other Transit', 'Radial Velocity', 'Other Methods']

    # set up all the legend objects
    items1 = [legitems[missions.index(ii)] for ii in topleg]
    items2 = [legitems[missions.index(ii)] for ii in bottomleg]
    items3 = [legitems[missions.index(ii)] for ii in vbottomleg]
    items4 = [solarleg]

    # create the two legends
    for ii in np.arange(4):
        if ii == 0:
            items = items4
        elif ii == 1:
            items = items3
        elif ii == 2:
            items = items2
        else:
            items = items1
        legend = Legend(items=items, location="center")

        if ii == 3:
            legend.title = 'Discovered by and Status'
            legend.spacing = 10
        else:
            legend.spacing = 11

        legend.location = (-70, 5)
        legend.label_text_align = 'left'
        legend.margin = 0

        fig.add_layout(legend, 'above')

    # overall figure title
    fig.title.text = 'All Confirmed and Candidate Planets'

    # create the four lines of credit text in the two bottom corners
    if ifig < 2:
        label_opts1 = dict(x=-85, y=32, x_units='screen', y_units='screen')
        label_opts2 = dict(x=-85, y=37, x_units='screen', y_units='screen')
        label_opts3 = dict(x=612, y=69, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')
        label_opts4 = dict(x=612, y=73, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')
    else:
        label_opts1 = dict(x=-85, y=32, x_units='screen', y_units='screen')
        label_opts2 = dict(x=-85, y=37, x_units='screen', y_units='screen')
        label_opts3 = dict(x=592, y=69, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')
        label_opts4 = dict(x=592, y=73, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')

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
    vertcent = {'margin': 'auto 5px'}
    button = Button(label="Download CSV of Selected Data",
                    button_type="primary", styles=vertcent)
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

    # allow for something to happen when you click on data points
    fig.add_tools(TapTool())
    # set up where to send people when they click on a planet
    uclick = CustomJS(args=dict(sources=sources), code=openurl)
    fig.js_on_event(Tap, uclick)

    # allow the user to choose the colors
    titlecent = {'margin': 'auto -3px auto 5px'}
    keplerpar = Div(text='Kepler:', styles=titlecent)
    keplercolor = ColorPicker(color=colors[0], width=60, height=30,
                              styles=vertcent)
    change_color(keplercolor, [glyphs[0], glyphs[5]])
    k2par = Div(text='K2:', styles=titlecent)
    k2color = ColorPicker(color=colors[1], width=60, height=30, styles=vertcent)
    change_color(k2color, [glyphs[1], glyphs[6]])
    tesspar = Div(text='TESS:', styles=titlecent)
    tesscolor = ColorPicker(color=colors[2], width=60, height=30,
                            styles=vertcent)
    change_color(tesscolor, [glyphs[2], glyphs[7]])
    confpar = Div(text='Other Transit:', styles=titlecent)
    confcolor = ColorPicker(color=colors[4], width=60, height=30,
                            styles=vertcent)
    change_color(confcolor, [glyphs[4]])
    rvpar = Div(text='RV:', styles=titlecent)
    rvcolor = ColorPicker(color=colors[3], width=60, height=30, styles=vertcent)
    change_color(rvcolor, [glyphs[3]])
    othpar = Div(text='Other:', styles=titlecent)
    othcolor = ColorPicker(color=colors[8], width=60, height=30,
                           styles=vertcent)
    change_color(othcolor, [glyphs[8]])

    if ifig in [0, 2]:
        jargs = dict(glyphs=glyphs, alphas=alphas, legends=legitems)
        fig.js_on_event('reset', CustomJS(args=jargs, code=reset))
        des = CustomJS(args=jargs, code=deselect)
        fig.js_on_event(SelectionGeometry, des)
        uns = CustomJS(args=jargs, code=unselect)
        fig.js_on_event(DoubleTap, uns)
        # for iglyph in glyphs:
        #     iglyph.js_on_change('visible', des)
        optrow = row(button, keplerpar, keplercolor, k2par, k2color, tesspar,
                     tesscolor, styles={'margin': '3px'})
        optrow2 = row(confpar, confcolor, rvpar, rvcolor, othpar, othcolor)
        layout = column(optrow, optrow2, fig)
    else:
        if ifig == 1:
            yrlabelopts = dict(x=520, y=410, x_units='screen', y_units='screen',
                               text_align='right', text_font_size='20pt',
                               text_baseline='top')
        else:
            yrlabelopts = dict(x=505, y=410, x_units='screen', y_units='screen',
                               text_align='right', text_font_size='20pt',
                               text_baseline='top')
        curyr = datetime.now().year
        yrtxt = f'{minyr}\u2013{curyr}'
        yrcap = Label(text=yrtxt, **yrlabelopts)
        fig.add_layout(yrcap)
        range_slider = RangeSlider(start=minyr, end=curyr, value=(minyr, curyr),
                                   step=1, title="Year Discovered", width=580,
                                   width_policy='fit')

        jargs = dict(glyphs=glyphs, alphas=alphas, legends=legitems,
                     slider=range_slider, label=yrcap)
        yrsel = CustomJS(args=jargs, code=yearselect)
        range_slider.js_on_change('value', yrsel)
        # even on a reset, reselect only the correct values rather than letting
        # everything get unselected
        rescode = "slider.value = [slider.start, slider.end];"
        fig.js_on_event('reset', CustomJS(args=jargs, code=rescode))
        sls = CustomJS(args=jargs, code=sliderselect)
        fig.js_on_event(SelectionGeometry, sls)
        fig.js_on_event(DoubleTap, yrsel)
        # for iglyph in glyphs:
        #     nw = CustomJS(args=jargs, code=nowvis)
        #     iglyph.js_on_change('visible', nw)

        playbutton = Toggle(label="\u25b6 Play", width=150, width_policy='fit',
                            button_type='success', styles=vertcent)
        playbutton.js_on_change('active', CustomJS(args=jargs, code=playpause))

        optrow = row(button, keplerpar, keplercolor, k2par, k2color, tesspar,
                     tesscolor, styles={'margin': '3px'})
        anirow = row(range_slider, playbutton, confpar, confcolor, rvpar,
                     rvcolor, othpar, othcolor)

        layout = column(optrow, anirow, fig)

    plotting.save(layout)

    plotting.show(layout)

    # save the individual pieces, so we can just embed the figure without the
    # whole html page
    script, div = components(layout, theme=theme)
    with open(embedfile, 'w') as ff:
        ff.write(script.strip())
        ff.write(div)
