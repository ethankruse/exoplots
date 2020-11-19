import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import SelectionGeometry
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, FuncTickFormatter, OpenURL, TapTool
from bokeh.models import CustomJS, Label, LassoSelectTool, Legend, LegendItem
from bokeh.models.widgets import Button
from bokeh.themes import Theme

from utils import csv_creation, get_update_time, log_axis_labels, reset
from utils import deselect, get_equilibrium_temperature, get_esm, get_tsm

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# what order to plot things and what the legend labels will say
status = ['Candidate', 'Confirmed', 'Confirmed with Mass Measurement']

# markers and colors in the same order as the missions above
markers = ['circle', 'square', 'triangle']
# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
colors = ['#228833', '#ee6677', '#aa3377']

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# calculate the values we want
albedo = 0.3
# fraction of planet radiating (1 for fast spin, 0.5 tidally locked)
redist = 0.5
rediststr = 'No Night-side Redistribution'

atmos = 'Atmos. \u03bc = 18'

dfpl['teq'] = get_equilibrium_temperature(dfpl, albedo=albedo,
                                          radiative_fraction=redist)

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
            ("Radius", "@radius{0,0[.][00]} Earth; @jupradius{0,0[.][0000]} Jup"),
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
            embedfile = f'_includes/{istr}_{ikey}_status_candidates_nolim_embed.html'
            fullfile = f'_includes/{istr}_{ikey}_status_candidates_nolim.html'

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
                                      tooltips=TOOLTIPS, plot_height=700)
            else:
                fig = plotting.figure(x_axis_type='log', y_axis_type='log',
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

            for ii, istat in enumerate(status):
                # select the appropriate set of planets for each mission
                # make the confirmed planets more opaque and bigger
                if istat == 'Confirmed':
                    good = (np.isfinite(dfpl['teq']) & np.isfinite(dfpl[ikey]) &
                            (dfpl['disposition'] == 'Confirmed') &
                            dfpl['flag_tran'] & #(dfpl['rade'] <= 2.) &
                            (dfpl[ikey] >= 0.) & (~np.isfinite(dfpl['masse'])) &
                            np.isfinite(dfpl['masse_est']))
                    alpha = 0.4
                    size = 7
                elif istat == 'Candidate':
                    good = (np.isfinite(dfpl['teq']) & np.isfinite(dfpl[ikey]) &
                            (dfpl['disposition'] == 'Candidate') &
                            dfpl['flag_tran'] & #(dfpl['rade'] <= 2.) &
                            (dfpl[ikey] >= 0.))
                    alpha = 0.4
                    size = 6
                else:
                    good = (np.isfinite(dfpl['teq']) & np.isfinite(dfpl[ikey]) &
                            (dfpl['disposition'] == 'Confirmed') &
                            dfpl['flag_tran'] & #(dfpl['rade'] <= 2.) &
                            (dfpl[ikey] >= 0.) & np.isfinite(dfpl['masse']))
                    alpha = 0.8
                    size = 8

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
                print(istat, ': ', good.sum())

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

                leg = LegendItem(label=istat + f' ({counts[ii]})',
                                 renderers=[glyph])
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

            # add the first y-axis's label and use our custom log formatting
            if imet == 'esm':
                fig.yaxis.axis_label = f'ESM ({refwavs[ifig]} micron)'
            else:
                fig.yaxis.axis_label = 'TSM'
            fig.yaxis.formatter = FuncTickFormatter(code=log_axis_labels())

            # add the x-axis's label and use our custom log formatting
            if ixx == 0:
                fig.xaxis.axis_label = 'Equilibrium Temp (K)'
            else:
                fig.xaxis.axis_label = 'Insolation (Earths)'
                fig.xaxis.formatter = FuncTickFormatter(code=log_axis_labels())
            # make high teqs on the left
            fig.x_range.flipped = True

            """
            # add the second y-axis's label
            if ifig == 0:
                fig.right[0].axis_label = 'Radius (Jupiter Radii)'
            """

            # which order to place the legend labels
            topleg = ['Candidate', 'Confirmed']
            bottomleg = ['Confirmed with Mass Measurement']

            # set up all the legend objects
            items1 = [legitems[status.index(ii)] for ii in topleg]
            items2 = [legitems[status.index(ii)] for ii in bottomleg]

            # create the two legends
            for ii in np.arange(2):
                if ii == 0:
                    items = items2
                else:
                    items = items1
                legend = Legend(items=items, location="center")

                if ii == 1:
                    legend.title = 'Planets and Candidates'
                    legend.spacing = 10
                else:
                    legend.spacing = 11

                """
                if ifig == 0:
                    legend.location = (-70, 5)
                else:
                """
                legend.location = (10, 5)
                legend.label_text_align = 'left'
                legend.margin = 0

                fig.add_layout(legend, 'above')

            # overall figure title
            fig.title.text = 'Transiting Planets and Planet Candidates'

            # create the four lines of credit text in the two bottom corners
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

            label_opts5 = dict(
                x=310, y=85,
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
            button = Button(label="Download CSV of Selected Data",
                            button_type="primary")
            # what is the header and what keys correspond to those columns for
            # output CSV files
            csvhead = '# Planet Name, Status, Period (days), Radius (Earths), ' \
                      'Radius (Jupiters), Insolation (Earths), Discovered by, ' \
                      f'T_eq (K), {imet.upper()}, Meas. Mass, Mass Est, {imag}'
            keys = ['planet', 'status', 'period', 'radius', 'jupradius',
                    'insolation', 'discovery', 'teq', 'met', 'mass', 'massest',
                    'mag']
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

            layout = column(button, fig)

            plotting.save(layout)

            plotting.show(layout)

            # save the individual pieces so we can just embed the figure
            # without the whole html page
            script, div = components(layout, theme=theme)
            with open(embedfile, 'w') as ff:
                ff.write(script)
                ff.write(div)
