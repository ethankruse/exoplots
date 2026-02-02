import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import SelectionGeometry, Tap
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BoxSelectTool, ColorPicker, CustomJS, Div, Label
from bokeh.models import LassoSelectTool, Legend, LegendItem, TapTool
from bokeh.models.widgets import Button
from bokeh.themes import Theme

from utils import change_color, csv_creation, deselect
from utils import get_equilibrium_temperature, get_esm, get_tsm
from utils import get_update_time, openurl, palette, reset


# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# in what order to plot things and what the legend labels will say
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

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# calculate the values we want
albedo = 0.3
# fraction of planet radiating (1 for fast spin, 0.5 tidally locked)
redist = 0.5
rediststr = 'No Night-side Redistribution'

atmos = 'Atmos. \u03bc = 18'

dfpl['teq'] = get_equilibrium_temperature(dfpl, albedo=albedo,
                                          radiative_fraction=redist).value

esmscale = 3.44

refwavs = [7.5, 15]
for refwav in refwavs:
    dfpl['esm'+str(refwav)] = get_esm(dfpl, wavelength_micron=refwav,
                                      scale=esmscale, albedo=albedo,
                                      radiative_fraction=redist)

tsmscale = 0.190

dfpl['tsm'] = get_tsm(dfpl, scale=tsmscale, albedo=albedo,
                      radiative_fraction=redist)

for fn, ifig in enumerate(np.arange(len(refwavs))):
    for imet in ['esm', 'tsm']:
        # multiple wavelengths only for ESM
        if imet == 'tsm' and fn > 0:
            continue

        if imet == 'esm':
            ikey = imet + str(refwavs[ifig])
            ttstr = 'ESM'
            imag = 'Kmag'
        else:
            ttstr = 'TSM'
            ikey = imet
            imag = 'Jmag'

        # what to display when hovering over a data point
        TOOLTIPS = [
            ("Planet", "@planet"),
            # only give the decimal and sig figs if needed
            ("Insolation", "@insolation{0,0[.][00]} Earths"),
            ("Equilibrium Temp", "@teq{0,0[.][0]} K"),
            (ttstr, "@met{0,0[.][00]}"),
            ("Radius", "@radius{0,0[.][00]} Earth; "
                       "@jupradius{0,0[.][0000]} Jup"),
            ("Period", "@period{0,0[.][0000]} days"),
            ("Measured Mass", "@mass{0,0[.][00]} Earth"),
            ("Estimated Mass", "@massest{0,0[.][00]} Earth"),
            ("Discovered by", "@discovery"),
            (imag, '@mag{0[.][00]}'),
            ("Status", "@status")
        ]

        for ixx in np.arange(2):
            if ixx == 0:
                istr = 'teq'
            else:
                istr = 'insol'

            # output files
            embedfile = f'_includes/{istr}_{ikey}_candidates_nolim_embed.html'
            fullfile = f'_includes/{istr}_{ikey}_candidates_nolim.html'

            # set up the full output file
            if ixx == 0:
                plotting.output_file(fullfile, title=f'Equilibrium Temp vs '
                                                     f'{imet.upper()}')
            else:
                plotting.output_file(fullfile, title=f'Insolation vs '
                                                     f'{imet.upper()}')

            # create the figure
            if ixx == 0:
                fig = plotting.figure(x_axis_type='linear', y_axis_type='log',
                                      tooltips=TOOLTIPS, height=700,
                                      width=750)
            else:
                fig = plotting.figure(x_axis_type='log', y_axis_type='log',
                                      tooltips=TOOLTIPS, height=700,
                                      width=750)

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
                    good = ((~np.isin(dfpl['facility'],
                                      ['Kepler', 'K2', 'TESS'])) &
                            np.isfinite(dfpl['teq']) & np.isfinite(dfpl[ikey]) &
                            (dfpl['disposition'] == 'Confirmed') &
                            dfpl['flag_tran'] &  # (dfpl['rade'] <= 2.) &
                            (dfpl[ikey] >= 0.))
                    alpha = 0.7
                    size = 8
                else:
                    fac = imiss.split()[0]
                    disp = imiss.split()[1]
                    good = ((dfpl['facility'] == fac) &
                            (dfpl['disposition'] == disp) &
                            np.isfinite(dfpl['teq']) & np.isfinite(dfpl[ikey]) &
                            dfpl['flag_tran'] &  # (dfpl['rade'] <= 2.) &
                            (dfpl[ikey] >= 0.))
                    if disp == 'Confirmed':
                        alpha = 0.7
                        size = 7
                    else:
                        # candidates get these values
                        alpha = 0.4
                        size = 6
                        # make the yellow darker
                        if fac == 'TESS':
                            alpha = 0.6

                counts.append(f'{good.sum():,}')

                # what the hover tooltip draws its values from
                source = plotting.ColumnDataSource(data=dict(
                        planet=dfpl['name'][good],
                        insolation=dfpl['insol'][good],
                        period=dfpl['period'][good],
                        radius=dfpl['rade'][good],
                        jupradius=dfpl['radj'][good],
                        host=dfpl['hostname'][good],
                        discovery=dfpl['facility'][good],
                        status=dfpl['disposition'][good],
                        url=dfpl['url'][good],
                        teq=dfpl['teq'][good],
                        met=dfpl[ikey][good],
                        mass=dfpl['masse'][good],
                        massest=dfpl['masse_est'][good],
                        mag=dfpl[imag][good]
                        ))
                print(imiss, ': ', good.sum())

                # plot the planets
                # nonselection stuff is needed to prevent planets in that
                # category from disappearing when you click on a data point
                # ("select" it)
                if ixx == 0:
                    glyph = fig.scatter('teq', 'met', color=colors[ii],
                                        source=source, size=size, alpha=alpha,
                                        marker=markers[ii], selection_alpha=0.8,
                                        nonselection_alpha=0.07,
                                        nonselection_color=colors[ii])
                else:
                    glyph = fig.scatter('insolation', 'met', color=colors[ii],
                                        source=source, size=size, alpha=alpha,
                                        marker=markers[ii], selection_alpha=0.8,
                                        nonselection_alpha=0.07,
                                        nonselection_color=colors[ii])
                glyphs.append(glyph)
                sources.append(source)
                alphas.append(alpha)
                # save the global min/max
                ymin = min(ymin, source.data['met'].min())
                ymax = max(ymax, source.data['met'].max())

                leg = LegendItem(label=imiss + f' ({counts[ii]})',
                                 renderers=[glyph])
                legitems.append(leg)

            # allow for something to happen when you click on data points
            fig.add_tools(TapTool())
            # set up where to send people when they click on a planet
            uclick = CustomJS(args=dict(sources=sources), code=openurl)
            fig.js_on_event(Tap, uclick)

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
            """
            if ifig == 0:
                fig.extra_y_ranges = {"jup": Range1d(start=ystart/radratio,
                                                     end=yend/radratio)}
                fig.add_layout(LogAxis(y_range_name="jup"), 'right')
            else:
                fig.y_range.start = ystart
                fig.y_range.end = yend
                fig.x_range.start = 40000
                fig.x_range.end = 0.1
            """
            if ymax < 100:
                fig.y_range.end = 110
            fig.y_range.start = 0.1

            # add the first y-axis's label
            if imet == 'esm':
                fig.yaxis.axis_label = f'\\[\\text{{ESM}} (\\mathrm{{' \
                                       f'{refwavs[ifig]} \\mu m}})\\]'
            else:
                fig.yaxis.axis_label = 'TSM'
            fig.yaxis.formatter.min_exponent = 3

            # add the x-axis's label
            if ixx == 0:
                fig.xaxis.axis_label = 'Equilibrium Temp (K)'
            else:
                fig.xaxis.axis_label = r'\[\text{Instellation} (\mathrm' \
                                       r'{F_\oplus})\]'
                fig.xaxis.formatter.min_exponent = 3
            # make high teqs on the left
            fig.x_range.flipped = True

            """
            # add the second y-axis's label
            if ifig == 0:
                fig.right[0].axis_label = 'Radius (Jupiter Radii)'
            """

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

                """
                if ifig == 0:
                    legend.location = (-70, 5)
                else:
                """
                legend.location = (-50, 5)
                legend.label_text_align = 'left'
                legend.margin = 0

                fig.add_layout(legend, 'above')

            # overall figure title
            fig.title.text = 'Transiting Planets and Planet Candidates'

            # create the four lines of credit text in the two bottom corners
            label_opts1 = dict(
                x=-65, y=32,
                x_units='screen', y_units='screen'
            )

            label_opts2 = dict(
                x=-65, y=37,
                x_units='screen', y_units='screen'
            )

            label_opts3 = dict(
                x=632, y=69,
                x_units='screen', y_units='screen', text_align='right',
                text_font_size='9pt'
            )

            label_opts4 = dict(
                x=632, y=73,
                x_units='screen', y_units='screen', text_align='right',
                text_font_size='9pt'
            )

            label_opts5 = dict(
                x=310, y=70,
                x_units='screen', y_units='screen', text_align='center'
            )

            msg1 = 'By Exoplots'
            # when did the data last get updated
            modtimestr = get_update_time().strftime('%Y %b %d')
            msg3 = 'Data: NASA Exoplanet Archive'
            msg4 = 'and ExoFOP-TESS'
            msg5 = f'Albedo = {albedo}; {rediststr}'
            if imet == 'tsm':
                msg5 += '; ' + atmos

            caption1 = Label(text=msg1, **label_opts1)
            caption2 = Label(text=modtimestr, **label_opts2)
            caption3 = Label(text=msg3, **label_opts3)
            caption4 = Label(text=msg4, **label_opts4)
            caption5 = Label(text=msg5, **label_opts5)

            fig.add_layout(caption1, 'below')
            fig.add_layout(caption2, 'below')
            fig.add_layout(caption3, 'below')
            fig.add_layout(caption4, 'below')
            fig.add_layout(caption5, 'below')

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
                                                  header=csvhead),
                                        code=csv_creation))

            # select multiple points to download
            box = BoxSelectTool()
            fig.add_tools(box)
            lasso = LassoSelectTool()
            fig.add_tools(lasso)

            des = CustomJS(args=dict(glyphs=glyphs, alphas=alphas,
                                     legends=legitems), code=deselect)
            fig.js_on_event(SelectionGeometry, des)
            fig.js_on_event('reset', CustomJS(args=dict(glyphs=glyphs,
                                                        alphas=alphas,
                                                        legends=legitems),
                                              code=reset))

            for iglyph in glyphs:
                iglyph.js_on_change('visible', des)

            # allow the user to choose the colors
            titlecent = {'margin': 'auto -3px auto 5px'}

            keplerpar = Div(text='Kepler:', styles=titlecent)
            keplercolor = ColorPicker(color=colors[0], width=60, height=30,
                                      styles=vertcent)
            change_color(keplercolor, glyphs[:2])
            k2par = Div(text='K2:', styles=titlecent)
            k2color = ColorPicker(color=colors[2], width=60, height=30,
                                  styles=vertcent)
            change_color(k2color, glyphs[2:4])
            tesspar = Div(text='TESS:', styles=titlecent)
            tesscolor = ColorPicker(color=colors[4], width=60, height=30,
                                    styles=vertcent)
            change_color(tesscolor, glyphs[4:7:2])
            confpar = Div(text='Other:', styles=titlecent)
            confcolor = ColorPicker(color=colors[5], width=60, height=30,
                                    styles=vertcent)
            change_color(confcolor, [glyphs[5]])

            optrow = row(button, keplerpar, keplercolor, k2par, k2color,
                         tesspar, tesscolor, confpar, confcolor,
                         styles={'margin': '3px'})
            layout = column(optrow, fig)

            plotting.save(layout)

            plotting.show(layout)

            # save the individual pieces, so we can just embed the figure
            # without the whole html page
            script, div = components(layout, theme=theme)
            with open(embedfile, 'w') as ff:
                ff.write(script.strip())
                ff.write(div)
