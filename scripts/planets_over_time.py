from datetime import datetime

import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColorPicker, Div, Label, NumeralTickFormatter
from bokeh.themes import Theme

from utils import change_color, get_update_time, palette

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
# XXX: can't figure out why the theme value overrides anything we set below
#   but only for this
theme._json['attrs']['Legend']['orientation'] = 'vertical'
curdoc().theme = theme

# in what order to plot things and what the legend labels will say
methods = ['Other', 'Radial Velocity', 'Transit']

# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
# colors = ['#228833', '#ee6677', '#ccbb44', '#aa3377', '#4477aa',
#           '#aaaaaa', '#66ccee']

colors = [palette['C2'], palette['C1'], palette['C0']]

# output files
embedfile_name = '_includes/per_year_{0}_embed.html'
fullfile_name = '_includes/per_year_{0}.html'

embedfilelog_name = '_includes/per_year_{0}_log_embed.html'
fullfilelog_name = '_includes/per_year_{0}_log.html'

embedfilecum_name = '_includes/per_year_{0}_cumul_embed.html'
fullfilecum_name = '_includes/per_year_{0}_cumul.html'

embedfilecumlog_name = '_includes/per_year_{0}_cumul_log_embed.html'
fullfilecumlog_name = '_includes/per_year_{0}_cumul_log.html'

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

years = np.arange(dfpl['year_discovered'].min(), datetime.now().year+1)

condata = {'years': years}
concumul = {'years': years}
conleglab = []
contots = []
concumtots = []

pcdata = {'years': years}
pccumul = {'years': years}
pcleglab = []
pctots = []
pccumtots = []

for ii, imeth in enumerate(methods):
    # select the appropriate set of planets for each mission
    if imeth == 'Other':
        good = ((~np.isin(dfpl['discoverymethod'], methods)) &
                (dfpl['disposition'] == 'Confirmed'))
    else:
        good = ((dfpl['discoverymethod'] == imeth) &
                (dfpl['disposition'] == 'Confirmed'))
    ntot = good.sum()

    base = []
    conll = []
    conisum = [0]
    pcll = []
    pcisum = [0]
    for iyear in years:
        ct = (dfpl['year_confirmed'][good] == iyear).sum()
        conll.append(ct)
        conisum.append(conisum[-1] + ct)
        base.append(0.01)

        pcct = (dfpl['year_discovered'][good] == iyear).sum()
        if imeth == 'Transit':
            canfind = dfpl['disposition'] == 'Candidate'
            pcct += (dfpl.loc[canfind, 'year_discovered'] == iyear).sum()

        pcll.append(pcct)
        pcisum.append(pcisum[-1] + pcct)

    if imeth == 'Transit':
        ntot += canfind.sum()

    conisum.pop(0)
    pcisum.pop(0)

    concumtots.append(conisum)
    contots.append(conll)
    condata[imeth] = conll
    concumul[imeth] = conisum
    if ii == 0:
        condata['base'] = base
        concumul['base'] = base
    conleglab.append(imeth + f' ({good.sum():,})')

    pccumtots.append(pcisum)
    pctots.append(pcll)
    pcdata[imeth] = pcll
    pccumul[imeth] = pcisum
    if ii == 0:
        pcdata['base'] = base
        pccumul['base'] = base
    pcleglab.append(imeth + f' ({ntot:,})')


contots = np.array(contots).sum(axis=0)
concumtots = np.array(concumtots).sum(axis=0)
condata['total'] = contots
concumul['total'] = concumtots

pctots = np.array(pctots).sum(axis=0)
pccumtots = np.array(pccumtots).sum(axis=0)
pcdata['total'] = pctots
pccumul['total'] = pccumtots

# get the exponential growth bit
cyear = get_update_time().year
# how many days is this year
fullyear = datetime(cyear + 1, 1, 1) - datetime(cyear, 1, 1)
# extrapolate this year's total through the full year
upscale = fullyear / (get_update_time() - datetime(cyear, 1, 1))
conscaled = concumtots * 1
conscaled[-1] = conscaled[-2] + upscale * (conscaled[-1] - conscaled[-2])
pcscaled = pccumtots * 1
pcscaled[-1] = pcscaled[-2] + upscale * (pcscaled[-1] - pcscaled[-2])
# use a weighted exponential growth fit
# see https://mathworld.wolfram.com/LeastSquaresFittingExponential.html
conexp = np.polyfit(np.arange(conscaled.size), np.log(conscaled),
                    1, w=np.log(conscaled))
conpreds = np.exp(np.polyval(conexp, np.arange(conscaled.size)))
contdouble = np.log(2) / conexp[0]
concumul['Predicted'] = conpreds

pcexp = np.polyfit(np.arange(pcscaled.size), np.log(pcscaled),
                   1, w=np.log(pcscaled))
pcpreds = np.exp(np.polyval(pcexp, np.arange(pcscaled.size)))
# to use the confirmed fit, just need to replace these two lines with the
# confirmed values above.
pctdouble = np.log(2) / pcexp[0]
pccumul['Predicted'] = pcpreds

fancytool0 = """
    <div>
        <span style="font-size: 12px; float:right;">@$name{0,0}</span>
        <span style="font-size: 12px; color: #5caddd; float:right;">
        @years $name:</span>          
    </div>
    <div>
        <span style="font-size: 12px; float:right;">@total{0,0}</span>
        <span style="font-size: 12px; color: #5caddd; float:right;">
        @years Total:</span> 
    </div>"""

fancytool1 = """
    <div>
        <span style="font-size: 12px; float:right;">@$name{0,0}</span>
        <span style="font-size: 12px; color: #5caddd; float:right;">
        $name through @years:</span>          
    </div>
    <div>
        <span style="font-size: 12px; float:right;">@total{0,0}</span>
        <span style="font-size: 12px; color: #5caddd; float:right;">
        Total through @years:</span> 
    </div>"""

# make the per year and then cumulative plots
for xx in np.arange(4):
    # set up the full output file and create the figure
    if (xx % 2) == 0:
        if xx == 0:
            txt = 'confirmed'
            tots = contots
            data = condata
            leglab = conleglab
            cumtots = concumtots
        else:
            txt = 'candidate'
            tots = pctots
            data = pcdata
            leglab = pcleglab
            cumtots = pccumtots
        plotting.output_file(fullfile_name.format(txt),
                             title='Planets Per Year')
        # '@years $name: @$name; Total: @total'
        fig = plotting.figure(tooltips=fancytool0, width=750,
                              height=600, y_range=(0, tots.max()*1.05))
        glyphs = fig.vbar_stack(methods, x='years', width=0.9, color=colors,
                                source=data, legend_label=leglab, line_width=0)
    else:
        if xx == 1:
            txt = 'confirmed'
            tots = contots
            data = condata
            leglab = conleglab
            cumtots = concumtots
            cumul = concumul
            tdouble = contdouble
        else:
            txt = 'candidate'
            tots = pctots
            data = pcdata
            leglab = pcleglab
            cumtots = pccumtots
            cumul = pccumul
            tdouble = pctdouble
        plotting.output_file(fullfilecum_name.format(txt),
                             title='Cumulative Planets')
        fig = plotting.figure(tooltips=fancytool1, width=750,
                              height=600, y_range=(0, cumtots.max()*1.05))
        # plot the exponential growth
        fig.line('years', 'Predicted', source=cumul, line_width=5,
                 line_color='black', name='Predicted',
                 legend_label=f'Doubling Time: {tdouble:.2f} years')
        glyphs = fig.vbar_stack(methods, x='years', width=0.9, color=colors,
                                source=cumul, legend_label=leglab, line_width=0)

    # add the first y-axis's label and use our custom log formatting
    # for both axes
    fig.yaxis.axis_label = 'Number'
    fig.yaxis.formatter = NumeralTickFormatter(format='0,0')

    # add the x-axis's label and use our custom log formatting
    if xx < 2:
        fig.xaxis.axis_label = 'Year of Confirmation'
    else:
        fig.xaxis.axis_label = 'Year of Discovery'

    # create the legend
    legend = fig.legend
    legend.location = 'top_left'
    # legend.orientation = "vertical"
    legend.title = 'Discovered via'
    # legend.spacing = 10
    # legend.margin = 8
    legend[0].items = legend[0].items[::-1]

    # overall figure title
    if xx == 0:
        fig.title.text = f'Confirmed Planets Per Year ({cumtots[-1]:,})'
    elif xx == 1:
        fig.title.text = f'Cumulative Confirmed Planets ({cumtots[-1]:,})'
    elif xx == 2:
        paren = f'({cumtots[-1]:,})'
        fig.title.text = 'Confirmed + Candidate Planets Per Year ' + paren
    else:
        paren = f'({cumtots[-1]:,})'
        fig.title.text = 'Cumulative Confirmed + Candidate Planets ' + paren
        fig.title.align = 'right'
    fig.title.text_font_size = '20pt'

    # create the three lines of credit text in the two bottom corners
    label_opts1 = dict(
        x=-84, y=32,
        x_units='screen', y_units='screen'
    )

    label_opts2 = dict(
        x=-84, y=37,
        x_units='screen', y_units='screen'
    )

    yup = 65

    label_opts3 = dict(
        x=612, y=yup,
        x_units='screen', y_units='screen', text_align='right',
        text_font_size='9pt'
    )

    label_opts4 = dict(
        x=612, y=yup+4,
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
    if xx > 1:
        fig.add_layout(caption4, 'below')

    # allow the user to choose the colors
    titlecent = {'margin': 'auto -3px auto 5px'}
    vertcent = {'margin': 'auto 5px'}
    keplerpar = Div(text='Transit:', styles=titlecent)
    keplercolor = ColorPicker(color=colors[2], width=60, height=30,
                              styles=vertcent)
    change_color(keplercolor, [glyphs[2]])
    k2par = Div(text='RV:', styles=titlecent)
    k2color = ColorPicker(color=colors[1], width=60, height=30, styles=vertcent)
    change_color(k2color, [glyphs[1]])
    tesspar = Div(text='Other:', styles=titlecent)
    tesscolor = ColorPicker(color=colors[0], width=60, height=30,
                            styles=vertcent)
    change_color(tesscolor, [glyphs[0]])

    optrow = row(keplerpar, keplercolor, k2par, k2color, tesspar,
                 tesscolor, styles={'margin': '3px'})
    layout = column(optrow, fig)

    plotting.show(layout)
    plotting.save(layout)

    # save the individual pieces, so we can just embed the figure without the
    # whole html page
    script, div = components(layout, theme=theme)
    if (xx % 2) == 0:
        with open(embedfile_name.format(txt), 'w') as ff:
            ff.write(script.strip())
            ff.write(div)
    else:
        with open(embedfilecum_name.format(txt), 'w') as ff:
            ff.write(script.strip())
            ff.write(div)

# now do the same thing but on log scale

methods.insert(0, 'base')
colors.insert(0, '#000000')
conleglab.insert(0, '')
pcleglab.insert(0, '')

# make the per year and then cumulative plots
for xx in np.arange(4):
    ymin = 0.8
    # set up the full output file and create the figure
    if (xx % 2) == 0:
        if xx == 0:
            txt = 'confirmed'
            tots = contots
            data = condata
            leglab = conleglab
            cumtots = concumtots
        else:
            txt = 'candidate'
            tots = pctots
            data = pcdata
            leglab = pcleglab
            cumtots = pccumtots
        ymax = 10.**(np.log10(tots.max()) +
                     0.05*(np.log10(tots.max()) - np.log10(ymin)))
        plotting.output_file(fullfilelog_name.format(txt),
                             title='Planets Per Year Log')
        fig2 = plotting.figure(tooltips=fancytool0, width=750,
                               height=600, y_range=(ymin, ymax),
                               y_axis_type='log')
        glyphs = fig2.vbar_stack(methods, x='years', width=0.9, color=colors,
                                 source=data, legend_label=leglab, line_width=0)
    else:
        if xx == 1:
            txt = 'confirmed'
            tots = contots
            data = condata
            leglab = conleglab
            cumtots = concumtots
            cumul = concumul
            tdouble = contdouble
        else:
            txt = 'candidate'
            tots = pctots
            data = pcdata
            leglab = pcleglab
            cumtots = pccumtots
            cumul = pccumul
            tdouble = pctdouble
        ymax = 10.**(np.log10(cumtots.max()) +
                     0.065*(np.log10(cumtots.max()) - np.log10(ymin)))
        plotting.output_file(fullfilecumlog_name.format(txt),
                             title='Planets Per Year Log')
        fig2 = plotting.figure(tooltips=fancytool1, width=750,
                               height=600, y_range=(ymin, ymax),
                               y_axis_type='log')
        # plot the exponential growth
        fig2.line('years', 'Predicted', source=cumul, line_width=5,
                  line_color='black', name='Predicted',
                  legend_label=f'Doubling Time: {tdouble:.2f} years')
        glyphs = fig2.vbar_stack(methods, x='years', width=0.9, color=colors,
                                 source=cumul, legend_label=leglab,
                                 line_width=0)

    # add the first y-axis's label
    fig2.yaxis.axis_label = 'Number'
    fig2.yaxis.formatter = NumeralTickFormatter(format='0,0')

    # add the x-axis's label
    if xx < 2:
        fig2.xaxis.axis_label = 'Year of Confirmation'
    else:
        fig2.xaxis.axis_label = 'Year of Discovery'

    # create the legend
    legend = fig2.legend
    legend.location = 'top_left'
    # legend.orientation = "vertical"
    legend.title = 'Discovered via'
    # legend.spacing = 10
    # legend.margin = 8

    # overall figure title
    if xx == 0:
        fig2.title.text = f'Confirmed Planets Per Year ({cumtots[-1]:,})'
    elif xx == 1:
        fig2.title.text = f'Cumulative Confirmed Planets ({cumtots[-1]:,})'
    elif xx == 2:
        paren = f'({cumtots[-1]:,})'
        fig2.title.text = 'Confirmed + Candidate Planets Per Year ' + paren
    else:
        paren = f'({cumtots[-1]:,})'
        fig2.title.text = 'Cumulative Confirmed + Candidate Planets ' + paren
        fig2.title.align = 'right'
    fig2.title.text_font_size = '20pt'

    if (xx % 2) == 1:
        legend[0].items.pop(1)
    else:
        legend[0].items.pop(0)
    legend[0].items = legend[0].items[::-1]

    # create the three lines of credit text in the two bottom corners
    label_opts1 = dict(
        x=-84, y=32,
        x_units='screen', y_units='screen'
    )

    label_opts2 = dict(
        x=-84, y=37,
        x_units='screen', y_units='screen'
    )

    yup = 65

    label_opts3 = dict(
        x=612, y=yup,
        x_units='screen', y_units='screen', text_align='right',
        text_font_size='9pt'
    )

    label_opts4 = dict(
        x=612, y=yup+4,
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

    fig2.add_layout(caption1, 'below')
    fig2.add_layout(caption2, 'below')
    fig2.add_layout(caption3, 'below')
    if xx > 1:
        fig2.add_layout(caption4, 'below')

    # allow the user to choose the colors
    titlecent = {'margin': 'auto -3px auto 5px'}
    vertcent = {'margin': 'auto 5px'}
    keplerpar = Div(text='Transit:', styles=titlecent)
    keplercolor = ColorPicker(color=colors[3], width=60, height=30,
                              styles=vertcent)
    change_color(keplercolor, [glyphs[3]])
    k2par = Div(text='RV:', styles=titlecent)
    k2color = ColorPicker(color=colors[2], width=60, height=30, styles=vertcent)
    change_color(k2color, [glyphs[2]])
    tesspar = Div(text='Other:', styles=titlecent)
    tesscolor = ColorPicker(color=colors[1], width=60, height=30,
                            styles=vertcent)
    change_color(tesscolor, [glyphs[1]])

    optrow = row(keplerpar, keplercolor, k2par, k2color, tesspar,
                 tesscolor, styles={'margin': '3px'})
    layout = column(optrow, fig2)

    plotting.show(layout)
    plotting.save(layout)

    # save the individual pieces, so we can just embed the figure without the
    # whole html page
    script, div = components(layout, theme=theme)
    if (xx % 2) == 0:
        with open(embedfilelog_name.format(txt), 'w') as ff:
            ff.write(script.strip())
            ff.write(div)
    else:
        with open(embedfilecumlog_name.format(txt), 'w') as ff:
            ff.write(script.strip())
            ff.write(div)
