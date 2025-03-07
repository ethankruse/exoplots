"""
Utility functions needed for every figure. Load and process the data in a
uniform way as well as run tests to ensure sensible results.
"""

from bokeh.models import ColorPicker

# order is green, salmon, gold, purple, blue, cyan
palette = {'C0': '#228833', 'C1': '#ee6677', 'C2': '#ccbb44', 'C3': '#aa3377',
           'C4': '#4477aa', 'C5': '#66ccee'}
# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/


class SolarSystem:
    def __init__(self):
        """
        Single location to store everything we need to put the Solar System into
        all the plots if we want.
        """
        import numpy as np
        from astropy import constants as const
        self.planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn',
                        'uranus', 'neptune']
        self.periods = np.array([87.97, 224.70, 365.26, 686.98, 4332.8, 10755.7,
                                 30687, 60190])
        self.radii = np.array([0.383, 0.95, 1.0, 0.53, 10.86, 9.00, 3.97, 3.86])
        self.mass = np.array([0.055, 0.815, 1.0, 0.107, 317.8, 95.159, 14.536,
                              17.147])
        self.semimajor = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537,
                                   19.19, 30.07])
        self.insolation = self.semimajor**-2

        # scale up Saturn so the actual planet is the same size as the rest
        self.width_mults = np.array([1, 1, 1, 1, 1, 100. / 57, 1, 1])
        self.urls = [f'https://raw.githubusercontent.com/ethankruse/exoplots/'
                     f'master/icons/{ipl}.png' for ipl in self.planets]
        # jupiter/earth radius ratio
        radratio = (const.R_jup / const.R_earth).value

        self.data = dict(planet=self.planets, period=self.periods,
                         radius=self.radii, jupradius=self.radii / radratio,
                         host=['Solar System'] * 8, discovery=['Earth'] * 8,
                         status=['Confirmed'] * 8, url=self.urls,
                         width_mult=self.width_mults, year=np.zeros(8),
                         insolation=self.insolation, semimajor=self.semimajor,
                         mass=self.mass)


def change_color(picker: ColorPicker, glyphs: list):
    """
    Link up the color tool to the glyphs we want it to control.

    Parameters
    ----------
    picker
    glyphs

    Returns
    -------

    """
    # all possible glyphs and what we need to update
    allglyph = ['glyph', 'selection_glyph', 'nonselection_glyph',
                'muted_glyph']
    allprops = ['fill_color', 'hatch_color', 'line_color']
    for iglyph in glyphs:
        for ag in allglyph:
            if ag not in vars(iglyph)['_property_values']:
                continue
            for prop in allprops:
                picker.js_link('color', vars(iglyph)['_property_values'][ag],
                               prop)
        for prop in allprops:
            if prop in vars(iglyph)['_property_values']:
                picker.js_link('color', iglyph, prop)


def get_update_time():
    """
    Return a datetime object representing the last time all the data files
    were generated.

    Returns
    -------
    datetime.datetime

    """
    import datetime
    dateloc = 'data/last_update_time.txt'
    with open(dateloc, 'r') as ff:
        lines = ff.readlines()
    return datetime.datetime.strptime(lines[0], '%Y-%m-%d %H:%M:%S.%f')


def get_equilibrium_temperature(df, albedo: float = 0.3,
                                radiative_fraction: float = 1):
    """
    Calculate the equilibrium temperature of a planet given the star's
    luminosity and the planet's distance away. Takes into account the planet's
    albedo and the fraction of the planet assumed to be heated and radiating
    at the equilibrium temperature.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe assumed to have columns of 'st_log_lum' and 'semi_au' that
        contain a star's log10 solar luminosity and the planet's semi-major
        axis in AU respectively. These are the only parameters needed to get
        an equilibrium temperature
    albedo : float, optional
        What fraction of a star's incoming light the planet reflects.
    radiative_fraction : float, optional
        What fraction of a planet is heated by the star and radiates at the
        returned equilibrium temperature. Full heat redistribution (entire
        planet is at the same temperature) has a fraction of 1; a tidally
        locked world where only half the planet is heated would be 0.5. The
        cold side is assumed to be negligible (temperature of 0 K).

    Returns
    -------
    astropy Quantity array
    """
    from astropy import constants as const
    import numpy as np

    if albedo < 0 or albedo > 1:
        raise Exception('Invalid albedo. Must be between 0 and 1.')
    if radiative_fraction < 0 or radiative_fraction > 1:
        raise Exception('Invalid radiative fraction. Must be between 0 and 1.')
    num = (10.**df['st_log_lum'].values) * const.L_sun * (1. - albedo)
    semisq = (df['semi_au'].values * const.au)**2
    denom = radiative_fraction * 16 * np.pi * const.sigma_sb * semisq
    return (num / denom)**0.25


def get_esm(df, wavelength_micron: float = 7.5, scale: float = 4.29, **kwargs):
    """
    Calculate the Emission Spectroscopy Metric for planets. The reference
    wavelength, overall scaling, and parameters for calculating the equilibrium
    temperature can all be changed.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe assumed to have columns needed to calculate equilibrium
        temperature and ESM, namely: 'st_log_lum', 'semi_au', 'st_teff',
        'tran_depth_ppm', and 'Kmag' that respectively contain a star's log10
        solar luminosity, the planet's semi-major axis in AU, stellar effective
        temperature, the planet's transit depth in ppm, and the star's K band
        magnitude.
    wavelength_micron : float, optional
        The wavelength to calculate the ESM at, in microns.
    scale : float, optional
        The overall ESM scaling, defaulting to the 4.29 used in
        Kempton et al., 2018.
    kwargs
        Any other parameters are passed to get_equilibrium_temperature.

    Returns
    -------
    astropy Quantity array
    """
    from astropy import constants as const
    from astropy import units as uu
    import numpy as np

    teq = get_equilibrium_temperature(df, **kwargs)
    exp1 = np.exp((const.h * const.c)/(wavelength_micron * uu.um * const.k_B *
                                       df['st_teff'].values * uu.K)) - 1
    exp2 = np.exp((const.h * const.c)/(wavelength_micron * uu.um * const.k_B *
                                       1.1 * teq)) - 1
    ecldep = df['tran_depth_ppm'].values * exp1 / exp2
    return scale * ecldep * (10.**(-0.2 * df['Kmag'].values))


def get_tsm(df, scale: float = 0.19, **kwargs):
    """
    Calculate the Transmission Spectroscopy Metric for planets. The
    overall S/N scaling value and parameters for calculating the equilibrium
    temperature can all be changed.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe assumed to have columns needed to calculate equilibrium
        temperature and TSM, namely: 'st_log_lum', 'semi_au', 'rade',
        'masse', 'masse_est', 'st_rad', and 'Jmag' that respectively contain
        a star's log10 solar luminosity, the planet's semi-major axis in AU,
        the planet's radius, the planet's measured mass, an estimate of
        the planet's mass from M-R relations when measured mass is unavailable,
        the star's radius, and the star's J band magnitude.
    scale : float, optional
        The overall TSM scaling, defaulting to the 0.19 used in
        Kempton et al., 2018 for terrestrial planets.
    kwargs
        Any other parameters are passed to get_equilibrium_temperature.

    Returns
    -------
    ndarray
    """
    import numpy as np
    teq = get_equilibrium_temperature(df, **kwargs).value

    num = scale * (df['rade']**3) * teq * (10.**(-0.2*df['Jmag']))

    combmass = df['masse_est'].values * 1
    isreal = np.isfinite(df['masse'])
    combmass[isreal] = df.loc[isreal, 'masse']

    denom = combmass * (df['st_rad']**2)

    return num / denom


def load_data(updated_koi_params=True, only_candidates=True):
    """
    Load our data tables and perform some data cleansing/updating to make them
    ready for use in our interactive figures.

    Parameters
    ----------
    updated_koi_params : bool
        If True, for all stars in the Kepler field, use the updated stellar and
        planet parameters from Berger 2023. Recalculate planet radii,
        insolations, etc. using these new Gaia assisted parameters.
    only_candidates : bool
        If True, don't update the confirmed planets with these KOI parameters.
        While most Kepler planets are just validated, the ones with masses and
        followup can have radii/densities messed up by mixing in these radii

    Returns
    -------
    dfcon : DataFrame
        All planets in the Exoplanet Archive confirmed planets table.
    dfkoi : DataFrame
        All planets in the Exoplanet Archive KOI planets table.
    dfk2 : DataFrame
        All planets in the Exoplanet Archive K2 planet candidates table.
    dftoi : DataFrame
        All planets in the ExoFOP-TESS planet candidates table.
    comp : DataFrame
        The composite data frame used in making all plots

    """
    import pandas as pd
    import numpy as np
    from astropy.coordinates import Angle
    from astropy import constants as const
    import warnings
    from glob import glob
    import os

    # load the data files
    datafile = 'data/confirmed-planets.csv'
    koifile = 'data/kepler-kois-full.csv'
    k2file = 'data/k2-candidates-table.csv'
    toifile = 'data/tess-candidates.csv'

    koidistfile = 'data/koi_distances.txt'
    k2distfile = 'data/k2oi_distances.txt'
    koistarsfile = 'data/GKTHCatalog_Table4.csv'
    koiplanetsfile = 'data/GKTHCatalog_Table5.csv'

    ticparams = 'data/full_tic.txt'

    dfcon = pd.read_csv(datafile, low_memory=False)

    dfkoi = pd.read_csv(koifile)
    dfk2 = pd.read_csv(k2file, low_memory=False)
    dftoi = pd.read_csv(toifile)
    if os.path.exists(ticparams):
        fulltic = pd.read_csv(ticparams)
    else:
        fulltic = None

    # what columns do we want/need in the final dataframe
    cols = ['name', 'hostname', 'IC', 'disposition', 'period', 'rade',
            'rade_err1', 'rade_err2', 'radj', 'radj_err1', 'radj_err2',
            'masse', 'masse_err1', 'masse_err2', 'massj', 'massj_err1',
            'massj_err2', 'tran_depth_ppm', 'tran_dur_hr', 'semi_au', 'eccen',
            'insol', 'distance_pc', 'year_discovered', 'year_confirmed',
            'discoverymethod', 'facility', 'st_mass', 'st_rad', 'st_teff',
            'st_log_lum', 'Jmag', 'Kmag', 'ra', 'dec', 'flag_tran',
            'flag_kepler', 'flag_k2', 'url']

    #########################
    # CONFIRMED PLANET PREP #
    #########################

    print('Handling confirmed planets.')
    # set our common names for all of these
    renames = {'pl_name': 'name', 'pl_orbper': 'period', 'pl_rade': 'rade',
               'pl_radeerr1': 'rade_err1', 'pl_radeerr2': 'rade_err2',
               'pl_radj': 'radj', 'pl_radjerr1': 'radj_err1',
               'pl_radjerr2': 'radj_err2', 'st_lum': 'st_log_lum',
               'pl_insol': 'insol', 'pl_orbsmax': 'semi_au',
               'pl_bmasse': 'masse', 'pl_bmasseerr1': 'masse_err1',
               'pl_bmasseerr2': 'masse_err2', 'pl_bmassj': 'massj',
               'pl_bmassjerr1': 'massj_err1', 'pl_bmassjerr2': 'massj_err2',
               'sy_kmag': 'Kmag', 'sy_jmag': 'Jmag',
               'pl_trandep': 'tran_depth_ppm', 'pl_trandur': 'tran_dur_hr',
               'pl_orbeccen': 'eccen', 'disc_facility': 'facility',
               'tran_flag': 'flag_tran', 'sy_dist': 'distance_pc',
               'disc_year': 'year_discovered'}
    dfcon.rename(columns=renames, inplace=True)
    # replace the long name with just TESS
    full = 'Transiting Exoplanet Survey Satellite (TESS)'
    dfcon['facility'] = dfcon['facility'].replace(full, 'TESS')

    dfcon['pl_bmasse_reflink'] = dfcon['pl_bmasse_reflink'].replace(np.nan, '')
    dfcon['pl_bmassj_reflink'] = dfcon['pl_bmassj_reflink'].replace(np.nan, '')
    dfcon['pl_rade_reflink'] = dfcon['pl_rade_reflink'].replace(np.nan, '')
    dfcon['pl_radj_reflink'] = dfcon['pl_radj_reflink'].replace(np.nan, '')

    # set all of these planets as confirmed
    dfcon['disposition'] = 'Confirmed'

    # where do we want to point people to on clicking?
    dfcon['url'] = ('https://exoplanetarchive.ipac.caltech.edu/overview/' +
                    dfcon['hostname'])

    # all transit flags are 0 or 1
    assert ((dfcon['flag_tran'] == 0) | (dfcon['flag_tran'] == 1)).all()
    # now make them bool flags as desired
    dfcon['flag_tran'] = dfcon['flag_tran'].astype(bool)

    # upper/lower limits are given values at that limit, and we need to remove
    # them for now
    dfcon.loc[dfcon['pl_orbperlim'] != 0, 'period'] = np.nan
    dfcon.loc[dfcon['pl_radelim'] != 0,
              ['rade', 'rade_err1', 'rade_err2']] = np.nan
    dfcon.loc[dfcon['pl_radjlim'] != 0,
              ['radj', 'radj_err1', 'radj_err2']] = np.nan
    dfcon.loc[dfcon['st_radlim'] != 0, 'st_rad'] = np.nan
    dfcon.loc[dfcon['st_masslim'] != 0, 'st_mass'] = np.nan
    dfcon.loc[dfcon['st_lumlim'] != 0, 'st_log_lum'] = np.nan
    dfcon.loc[dfcon['st_tefflim'] != 0, 'st_teff'] = np.nan
    dfcon.loc[dfcon['pl_insollim'] != 0, 'insol'] = np.nan
    dfcon.loc[dfcon['pl_orbsmaxlim'] != 0, 'semi_au'] = np.nan
    dfcon.loc[dfcon['pl_ratdorlim'] != 0, 'pl_ratdor'] = np.nan
    dfcon.loc[dfcon['pl_bmasselim'] != 0,
              ['masse', 'masse_err1', 'masse_err2']] = np.nan
    dfcon.loc[dfcon['pl_bmassjlim'] != 0,
              ['massj', 'massj_err1', 'massj_err2']] = np.nan
    dfcon.loc[dfcon['pl_trandeplim'] != 0, 'tran_depth_ppm'] = np.nan
    dfcon.loc[dfcon['pl_trandurlim'] != 0, 'tran_dur_hr'] = np.nan

    ct = 0
    # 2 planets (OGLE-TR-111 b, Kepler-49 b) have different references
    # for the two masses
    # XXX: notify archive
    for ii in np.arange(dfcon['masse'].size):
        r1 = dfcon.at[ii, 'pl_bmasse_reflink']
        r2 = dfcon.at[ii, 'pl_bmassj_reflink']
        if r1 != r2:
            ct += 1
    assert ct == 1

    # both always exist or not together
    badm = (np.isfinite(dfcon['masse']) ^ np.isfinite(dfcon['massj']))
    assert badm.sum() == 0
    badm = (np.isfinite(dfcon['masse_err1']) ^ np.isfinite(dfcon['massj_err1']))
    assert badm.sum() == 0
    badm = (np.isfinite(dfcon['masse_err2']) ^ np.isfinite(dfcon['massj_err2']))
    assert badm.sum() == 0

    # remove calculated values from true masses. we'll make a calculated
    # column later
    badme = dfcon['pl_bmasse_reflink'].str.contains('Calculated')
    badmj = dfcon['pl_bmassj_reflink'].str.contains('Calculated')
    assert not (badme ^ badmj).any()
    dfcon.loc[badme, ['masse', 'massj', 'masse_err1', 'masse_err2',
                      'massj_err1', 'massj_err2']] = np.nan

    massrat = (const.M_jup/const.M_earth).value
    # because earth and jup masses don't always agree, make them
    # uniform and treat Earth as truth
    dfcon['massj'] = dfcon['masse'] / massrat
    dfcon['massj_err1'] = dfcon['masse_err1'] / massrat
    dfcon['massj_err2'] = dfcon['masse_err2'] / massrat

    # avoid a defrag warning without dealing with cleaning up the code
    dfcon = dfcon.copy()

    ct = 0
    for ii in np.arange(dfcon['masse'].size):
        r1 = dfcon.at[ii, 'pl_rade_reflink']
        r2 = dfcon.at[ii, 'pl_radj_reflink']
        if r1 != r2:
            ct += 1
    assert ct == 0
    # jupiter/earth radius ratio
    radratio = (const.R_jup/const.R_earth).value
    badrj = (np.isfinite(dfcon['radj']) ^ np.isfinite(dfcon['rade']))
    assert badrj.sum() == 0
    badrj1 = (np.isfinite(dfcon['radj_err1']) ^ np.isfinite(dfcon['rade_err1']))
    assert badrj1.sum() == 0
    badrj2 = (np.isfinite(dfcon['radj_err2']) ^ np.isfinite(dfcon['rade_err2']))
    assert badrj2.sum() == 0
    assert np.allclose(badrj, badrj1) and np.allclose(badrj, badrj2)

    # remove calculated values from true radii. we'll make a calculated
    # column later
    badre = dfcon['pl_rade_reflink'].str.contains('Calculated')
    badrj = dfcon['pl_radj_reflink'].str.contains('Calculated')
    assert not (badre ^ badrj).any()
    dfcon.loc[badre, ['rade', 'radj', 'rade_err1', 'rade_err2',
                      'radj_err1', 'radj_err2']] = np.nan

    # because earth and jup radii use different sources, make them
    # uniform and treat Earth as truth
    dfcon['radj'] = dfcon['rade'] / radratio
    dfcon['radj_err1'] = dfcon['rade_err1'] / radratio
    dfcon['radj_err2'] = dfcon['rade_err2'] / radratio

    # convert their depth in % to depth in ppm
    dfcon['tran_depth_ppm'] *= 1e4
    # we should have transit depths for these, the papers just reported
    # only radii and not depths, so guesstimate
    getdep = (dfcon['flag_tran'] & (~np.isfinite(dfcon['tran_depth_ppm'])) &
              np.isfinite(dfcon['rade']) & np.isfinite(dfcon['st_rad']))

    sunearth = (const.R_sun/const.R_earth).value
    tranrat = dfcon['rade']**2 / (dfcon['st_rad'] * sunearth)**2
    tranrat *= 1e6
    dfcon.loc[getdep, 'tran_depth_ppm'] = tranrat[getdep]

    # for some reason these claim to have transits but either no
    # stellar or planet radius measurement
    getrad = (dfcon['flag_tran'] & np.isfinite(dfcon['tran_depth_ppm']) &
              ((~np.isfinite(dfcon['rade'])) | (~np.isfinite(dfcon['st_rad']))))
    assert getrad.sum() == 0

    # set whether these were observed by Kepler or K2
    dfcon['flag_kepler'] = False
    # these were labeled in the old table as being in the Kepler field
    # but didn't have default host names indicating such
    others = ['PH1', 'PH2', 'TrES-2', 'HAT-P-7', 'HAT-P-11',
              '2MASS J19383260+4603591']
    for ind, icon in dfcon.iterrows():
        iskep = (icon['hostname'][:3] in ['KOI', 'Kep', 'KIC'] or
                 icon['hostname'] in others)
        if iskep:
            dfcon.loc[ind, 'flag_kepler'] = True

    dfcon['flag_k2'] = False
    # these were labeled in the old table as being in the K2 fields
    # but didn't have default host names indicating such
    others = ['BD+20 594', 'G 9-40', 'GJ 9827', 'HAT-P-56', 'HATS-11',
              'HATS-12', 'HATS-36', 'HATS-9', 'HD 106315', 'HD 3167',
              'HD 72490', 'HD 80653', 'HD 89345', 'HIP 116454',
              'HIP 41378', 'Qatar-2', 'Ross 128', 'TRAPPIST-1',
              'V1298 Tau', 'WASP-151', 'WASP-157', 'WASP-28', 'WASP-47',
              'WASP-75', 'WASP-85 A', 'Wolf 503']
    for ind, icon in dfcon.iterrows():
        isk2 = (icon['hostname'][:2] == 'K2' or
                icon['hostname'][:4] == 'EPIC' or
                icon['hostname'] in others)
        if isk2:
            dfcon.loc[ind, 'flag_k2'] = True

    # fill in any missing luminosities with our own calculation
    # (archive claims they already do this)
    tmplums = (dfcon['st_rad']**2) * ((dfcon['st_teff'] / 5772)**4)
    toadd = (~np.isfinite(dfcon['st_log_lum'])) & np.isfinite(tmplums)
    assert toadd.sum() == 0

    # fill in any missing semi-major axes from Kepler's third law first
    tmpau = (((dfcon['period'] / 365.25)**2) * dfcon['st_mass'])**(1./3.)
    repau = (~np.isfinite(dfcon['semi_au'])) & np.isfinite(tmpau)
    dfcon.loc[repau, 'semi_au'] = tmpau[repau]

    # then fill in any missing semi-major axes with a/R* * R*
    # convert to AU; 1 AU = 215 Rsun
    tmpau2 = dfcon['pl_ratdor'] * dfcon['st_rad'] / (const.au/const.R_sun).value
    repau2 = (~np.isfinite(dfcon['semi_au'])) & np.isfinite(tmpau2)
    # this so far is only necessary for HIP 41378 d
    assert repau2.sum() == 1
    dfcon.loc[repau2, 'semi_au'] = tmpau2[repau2]

    # calculate insolations ourselves and fill in any missing that we can
    tmpinsol = (10.**dfcon['st_log_lum']) * (dfcon['semi_au']**-2)
    repinsol = (~np.isfinite(dfcon['insol'])) & np.isfinite(tmpinsol)
    dfcon.loc[repinsol, 'insol'] = tmpinsol[repinsol]

    # confirmed planets don't have Input Catalog numbers, so set them to nan
    dfcon['IC'] = float('nan')

    # make a confirmation year column as well
    dfcon['year_confirmed'] = dfcon['year_discovered']

    # do all the tests to make sure things are working

    # everything has a facility name
    assert (np.array([len(xi) for xi in dfcon['facility']]) > 0).all()
    assert (dfcon['disposition'] == 'Confirmed').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in dfcon['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in dfcon['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in dfcon['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert not np.isfinite(dfcon['IC']).any()
    assert (np.array([len(xi) for xi in dfcon['discoverymethod']]) > 0).all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(dfcon['distance_pc']) |
            (dfcon['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(dfcon['st_rad']) | (dfcon['st_rad'] > 0)).all()
    assert (~np.isfinite(dfcon['st_mass']) | (dfcon['st_mass'] > 0)).all()
    assert (~np.isfinite(dfcon['st_teff']) | (dfcon['st_teff'] > 100)).all()
    assert (~np.isfinite(dfcon['st_log_lum']) |
            ((dfcon['st_log_lum'] > -8) & (dfcon['st_log_lum'] < 5))).all()
    assert (~np.isfinite(dfcon['Kmag']) | (dfcon['Kmag'] > -5)).all()
    assert (~np.isfinite(dfcon['Jmag']) | (dfcon['Jmag'] > -5)).all()

    # RA and Dec are both valid
    assert ((dfcon['ra'] >= 0) & (dfcon['ra'] <= 360.)).all()
    assert ((dfcon['dec'] >= -90) & (dfcon['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert (~np.isfinite(dfcon['period']) | (dfcon['period'] > 0)).all()
    assert (~np.isfinite(dfcon['semi_au']) | (dfcon['semi_au'] > 0)).all()
    # XXX: while they're truncating at 2 decimal places, BD-21 397 c is broken
    good = (~np.isfinite(dfcon['insol']) | (dfcon['insol'] > 0))
    # assert (~np.isfinite(dfcon['insol']) | (dfcon['insol'] > 0)).all()
    assert (~good).sum() == 1
    assert (~np.isfinite(dfcon['rade']) | (dfcon['rade'] > 0)).all()
    assert (~np.isfinite(dfcon['rade_err1']) | (dfcon['rade_err1'] >= 0)).all()
    assert (~np.isfinite(dfcon['rade_err2']) | (dfcon['rade_err2'] <= 0)).all()
    assert (~np.isfinite(dfcon['radj']) | (dfcon['radj'] > 0)).all()
    assert (~np.isfinite(dfcon['radj_err1']) | (dfcon['radj_err1'] >= 0)).all()
    assert (~np.isfinite(dfcon['radj_err2']) | (dfcon['radj_err2'] <= 0)).all()
    assert (~np.isfinite(dfcon['masse']) | (dfcon['masse'] > 0)).all()
    assert (~np.isfinite(dfcon['masse_err1']) |
            (dfcon['masse_err1'] >= 0)).all()
    assert (~np.isfinite(dfcon['masse_err2']) |
            (dfcon['masse_err2'] <= 0)).all()
    assert (~np.isfinite(dfcon['massj']) | (dfcon['massj'] > 0)).all()
    assert (~np.isfinite(dfcon['massj_err1']) |
            (dfcon['massj_err1'] >= 0)).all()
    assert (~np.isfinite(dfcon['massj_err2']) |
            (dfcon['massj_err2'] <= 0)).all()
    assert (~np.isfinite(dfcon['tran_depth_ppm']) |
            (dfcon['tran_depth_ppm'] > 0)).all()
    assert (~np.isfinite(dfcon['tran_dur_hr']) |
            (dfcon['tran_dur_hr'] > 0)).all()

    # Jup and Earth radii and masses are either defined or not together
    assert np.allclose(np.isfinite(dfcon['radj']), np.isfinite(dfcon['rade']))
    assert np.allclose(np.isfinite(dfcon['radj_err1']),
                       np.isfinite(dfcon['rade_err1']))
    assert np.allclose(np.isfinite(dfcon['radj_err2']),
                       np.isfinite(dfcon['rade_err2']))
    assert ((~np.isfinite(dfcon['rade_err1'])) | (dfcon['rade_err1'] == 0) |
            ((dfcon['rade_err1'] / dfcon['radj_err1'] > 0.99 * radratio) &
             (dfcon['rade_err1'] / dfcon['radj_err1'] < 1.01 * radratio))).all()
    assert ((~np.isfinite(dfcon['rade_err2'])) | (dfcon['rade_err2'] == 0) |
            ((dfcon['rade_err2'] / dfcon['radj_err2'] > 0.99 * radratio) &
             (dfcon['rade_err2'] / dfcon['radj_err2'] < 1.01 * radratio))).all()
    assert ((~np.isfinite(dfcon['rade'])) |
            ((dfcon['rade'] / dfcon['radj'] > 0.99 * radratio) &
             (dfcon['rade'] / dfcon['radj'] < 1.01 * radratio))).all()
    assert np.allclose(np.isfinite(dfcon['massj']),
                       np.isfinite(dfcon['masse']))
    assert np.allclose(np.isfinite(dfcon['massj_err1']),
                       np.isfinite(dfcon['masse_err1']))
    assert np.allclose(np.isfinite(dfcon['massj_err2']),
                       np.isfinite(dfcon['masse_err2']))
    assert ((~np.isfinite(dfcon['masse'])) |
            ((dfcon['masse'] / dfcon['massj'] > 0.99 * massrat) &
             (dfcon['masse'] / dfcon['massj'] < 1.01 * massrat))).all()
    assert ((~np.isfinite(dfcon['masse_err1'])) | (dfcon['masse_err1'] == 0) |
            ((dfcon['masse_err1'] / dfcon['massj_err1'] > 0.99*massrat) &
             (dfcon['masse_err1'] / dfcon['massj_err1'] < 1.01*massrat))).all()
    assert ((~np.isfinite(dfcon['masse_err2'])) | (dfcon['masse_err2'] == 0) |
            ((dfcon['masse_err2'] / dfcon['massj_err2'] > 0.99*massrat) &
             (dfcon['masse_err2'] / dfcon['massj_err2'] < 1.01*massrat))).all()
    # these flags at least have the right number of good values
    assert dfcon['flag_kepler'].sum() > 2000
    assert dfcon['flag_k2'].sum() > 400
    assert dfcon['flag_tran'].sum() > 3000

    # discovery and confirmation years make sense
    assert (dfcon['year_discovered'] >= 1989).all()
    assert np.allclose(dfcon['year_confirmed'], dfcon['year_discovered'])

    assert (~np.isfinite(dfcon['eccen']) |
            ((dfcon['eccen'] >= 0) & (dfcon['eccen'] < 1))).all()

    # create the composite, single data frame for all the planets and
    # planet candidates
    comp = dfcon[cols].copy()

    #################
    # KOI LIST PREP #
    #################

    print('Handling KOIs.')
    # make these not all caps
    dfkoi['disposition'] = dfkoi['koi_disposition'].str.title()
    assert np.unique(dfkoi['disposition']).size == 3

    # set our common names for all of these
    renames = {'koi_period': 'period', 'koi_prad': 'rade',
               'koi_prad_err1': 'rade_err1', 'koi_prad_err2': 'rade_err2',
               'kepid': 'IC', 'koi_insol': 'insol', 'koi_sma': 'semi_au',
               'koi_smass': 'st_mass', 'koi_srad': 'st_rad',
               'koi_steff': 'st_teff', 'kepoi_name': 'name',
               'koi_kmag': 'Kmag', 'koi_depth': 'tran_depth_ppm',
               'koi_duration': 'tran_dur_hr', 'koi_jmag': 'Jmag'}
    dfkoi.rename(columns=renames, inplace=True)

    # make KOI strings into the format we expect
    dfkoi['name'] = dfkoi['name'].replace(to_replace='K0+', value='KOI-',
                                          regex=True)

    assert (~np.isfinite(dfkoi['insol']) | (dfkoi['insol'] >= 0)).all()
    assert (~np.isfinite(dfkoi['semi_au']) | (dfkoi['semi_au'] > 0)).all()
    assert (~np.isfinite(dfkoi['st_rad']) | (dfkoi['st_rad'] > 0)).all()
    assert (~np.isfinite(dfkoi['st_mass']) | (dfkoi['st_mass'] >= 0)).all()
    assert (~np.isfinite(dfkoi['st_teff']) | (dfkoi['st_teff'] > 100)).all()
    assert (dfkoi['period'] > 0).all()

    # go through and assign KOIs the year they first showed up in a KOI
    # catalog. We're assuming in this process that a particular KOI number will
    # always refer to the same planet.
    dfkoi['year_discovered'] = 1990

    # these first 2 KOI tables aren't archived on Exoplanet Archive
    earlykois = ['data/koi1.txt', 'data/koi2.txt']
    allkois = glob('data/kepler-kois-q*')
    allkois.sort()
    # year the KOI tables were published
    koiyears = [2013, 2014, 2015, 2015, 2016, 2018]

    # load the two early KOI tables. Just use KOI names
    k1 = np.loadtxt(earlykois[0], dtype='<U12', usecols=(0,))
    for ii in np.arange(k1.size):
        k1[ii] = 'KOI-' + k1[ii]
    k2 = np.loadtxt(earlykois[1], dtype='<U12', usecols=(0,), skiprows=73)
    for ii in np.arange(k2.size):
        k2[ii] = 'KOI-' + k2[ii]

    # load the archived KOI tables
    dfs = []
    for ifile in allkois:
        df = pd.read_csv(ifile)
        df['kepoi_name'] = df['kepoi_name'].replace(to_replace='K0+',
                                                    value='KOI-', regex=True)
        dfs.append(df)

    # find the first time a particular KOI number is mentioned and set its year
    for index, row in dfkoi.iterrows():
        ikoi = row['name']
        if ikoi in k1 or ikoi in k2:
            dfkoi.at[index, 'year_discovered'] = 2011
            continue
        for ii, df in enumerate(dfs):
            if ikoi in df['kepoi_name'].values:
                dfkoi.at[index, 'year_discovered'] = koiyears[ii]
                break

    # we got them all
    assert (dfkoi['year_discovered'].min() == 2011 and
            dfkoi['year_discovered'].max() == 2018)

    # give KOIs units of Jupiter radii
    dfkoi['radj'] = dfkoi['rade'] / radratio
    dfkoi['radj_err1'] = dfkoi['rade_err1'] / radratio
    dfkoi['radj_err2'] = dfkoi['rade_err2'] / radratio

    # set the appropriate discovery facility for candidates
    dfkoi['facility'] = 'Kepler'

    # where do we want to point people to on clicking?
    exo = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Display' \
          'Overview/nph-DisplayOverview?objname='
    dfkoi['url'] = (exo + dfkoi['name'].str.slice(0, -3) +
                    '&type=KEPLER_TCE_HOST')

    # KOI-1101.02 is a known duplicate of 1101.01. Remove it.
    assert len(dfkoi[dfkoi['name'] == 'KOI-1101.02']) == 1
    dfkoi.drop(dfkoi[dfkoi['name'] == 'KOI-1101.02'].index, inplace=True)
    assert len(dfkoi[dfkoi['name'] == 'KOI-1101.02']) == 0

    # these were removed as FPs from the confirmed talbe, but not the KOI list
    # XXX: notify archive
    newfps = ['Kepler-699 b', 'Kepler-840 b', 'Kepler-854 b', 'Kepler-486 b',
              'Kepler-492 b', 'Kepler-706 b', 'Kepler-470 b', 'Kepler-494 b',
              'Kepler-628 b', 'Kepler-807 b', 'KIC 5951458 b', 'Kepler-488 b']
    badrows = np.isin(dfkoi['kepler_name'], newfps)
    assert (dfkoi.loc[badrows, 'disposition'] == 'Confirmed').all()
    dfkoi.loc[badrows, 'disposition'] = 'False Positive'

    # there's not an easy way to tie confirmed planets in the KOI table to
    # entries in the confirmed planets table. instead, match by RA/Dec/Period
    koicon = dfkoi['disposition'] == 'Confirmed'
    koican = dfkoi['disposition'] == 'Candidate'

    # KOI-4441, 4777, 5475 were KOIs at half the period of the confirmed planet
    # and 5568 a KOI at 1/3 the confirmed period. KOI-2174.03 confirmed at
    # double the period
    excluded = ['KOI-4441.01', 'KOI-5568.01', 'KOI-5475.01', 'KOI-2174.03',
                'KOI-4777.01']
    # what the name is in the confirmed planets table
    real = ['Kepler-1604 b', 'Kepler-1633 b', 'Kepler-1632 b', 'Kepler-1802 b',
            'KOI-4777.01']
    fixed = np.zeros(len(excluded), dtype=bool)

    # make sure all confirmed KOIs are in the confirmed table exactly once
    for index, icon in dfkoi[koicon].iterrows():
        res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - icon['period']) /
                               icon['period']) < 0.01))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can match up manually
            assert icon['name'] in excluded
            rname = real[excluded.index(icon['name'])]
            res = np.where(comp['name'] == rname)[0]
            assert len(res) == 1
            fixed[excluded.index(icon['name'])] = True
        # update and sync the discovery year in both tables
        res = res[0]
        myr = min(comp.at[res, 'year_discovered'], icon['year_discovered'])
        comp.at[res, 'year_discovered'] = myr
        dfkoi.at[index, 'year_discovered'] = myr

    assert fixed.all()

    # these are now confirmed, but they didn't update it
    # in the KOI table
    newconf = []
    newconf2 = []
    isconf = np.zeros(len(newconf), dtype=bool)

    # make sure all candidate KOIs aren't in the confirmed table
    for index, ican in dfkoi[koican].iterrows():
        res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - ican['period']) /
                               ican['period']) < 0.01))
        res = res[0]
        if len(res) > 0:
            assert len(res) == 1
            assert ican['name'] in newconf
            dfkoi.loc[index, 'disposition'] = 'Confirmed'
            short = newconf2[newconf.index(ican['name'])]
            dfkoi.loc[index, 'kepler_name'] = short
            isconf[newconf.index(ican['name'])] = True
            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], ican['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dfkoi.at[index, 'year_discovered'] = myr

    assert isconf.all()

    # make the KIC the host name here
    dfkoi['hostname'] = 'KIC ' + dfkoi['IC'].astype(str)

    # set up a distance field that is the same in all 4 groups
    koidists = np.zeros(dfkoi['IC'].size)

    kics, k1dists = np.loadtxt(koidistfile, unpack=True)
    kics = kics.astype(int)
    for ii, ikoi in enumerate(dfkoi['IC']):
        srch = np.where(kics == ikoi)[0]
        if len(srch) == 1:
            koidists[ii] = k1dists[srch[0]]
        elif len(srch) == 0:
            warnings.warn(f'Can not find distance for KIC {ikoi}')
            koidists[ii] = np.nan
        else:
            raise Exception(f'Multiple distances for KIC {ikoi}?')
    dfkoi['distance_pc'] = koidists

    # fill in missing luminosities with our own calculation
    tmplums = (dfkoi['st_rad'] ** 2) * ((dfkoi['st_teff'] / 5772) ** 4)
    dfkoi['st_log_lum'] = np.log10(tmplums)

    # KOI insolations only go to 2 sig figs so gets the distant ones wrong
    # so calculate our own in a consistent way
    tmpau = (((dfkoi['period'] / 365.25)**2) * dfkoi['st_mass'])**(1./3.)
    dfkoi.loc[np.isfinite(tmpau), 'semi_au'] = tmpau[np.isfinite(tmpau)]

    tmpinsol = (10.**dfkoi['st_log_lum']) * (dfkoi['semi_au']**-2)
    dfkoi.loc[np.isfinite(tmpinsol), 'insol'] = tmpinsol[np.isfinite(tmpinsol)]

    if updated_koi_params:
        # load the Berger 2023 data
        koistars = pd.read_csv(koistarsfile)
        koiplanets = pd.read_csv(koiplanetsfile)

        koiplanets['id_planet'] = koiplanets['id_planet'].replace(to_replace='K0+', value='KOI-', regex=True)

        koicon = dfkoi['disposition'] == 'Confirmed'
        koican = dfkoi['disposition'] == 'Candidate'

        # keep track of which planets in the confirmed list don't have a KOI,
        # so we have to find its KIC/get stellar parameters a fancier way
        cononly = comp['flag_kepler'] & True

        # match the confirmed KOIs to the appropriate confirmed planets and
        # get that KOIs new stellar values
        for index, icon in dfkoi[koicon].iterrows():
            res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                           (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                           (np.abs((comp['period'] - icon['period']) /
                                   icon['period']) < 0.01))
            res = res[0]
            if len(res) != 1:
                # special cases I know about that we can match up manually
                assert icon['name'] in excluded
                rname = real[excluded.index(icon['name'])]
                res = np.where(comp['name'] == rname)[0]
                assert len(res) == 1

            res = res[0]
            cononly[res] = False

            fdp = np.where(koiplanets['id_planet'] == icon['name'])[0]
            fds = np.where(koistars['id_starname'] == f'kic{icon["IC"]}')[0]
            # at least update the distance with DR3/TIC values
            if len(fdp) != 1 or len(fds) != 1:
                comp.at[res, 'distance_pc'] = icon['distance_pc']
            if not only_candidates:
                # only care about updating the confirmed table
                comp.at[res, 'st_mass'] = koistars.loc[fds, 'iso_mass']
                comp.at[res, 'st_rad'] = koistars.loc[fds, 'iso_rad']
                comp.at[res, 'st_teff'] = koistars.loc[fds, 'iso_teff']
                l10 = np.log10(koistars.loc[fds, 'iso_lum'])
                comp.at[res, 'st_log_lum'] = l10
                comp.at[res, 'distance_pc'] = koistars.loc[fds, 'iso_dis']

                comp.at[res, 'semi_au'] = koiplanets.loc[fdp, 'sma']
                comp.at[res, 'insol'] = koiplanets.loc[fdp, 'insol']
                comp.at[res, 'rade'] = koiplanets.loc[fdp, 'prad']
                comp.at[res, 'rade_err1'] = koiplanets.loc[fdp, 'prad_err1']
                comp.at[res, 'rade_err2'] = koiplanets.loc[fdp, 'prad_err2']
                comp.at[res, 'radj'] = koiplanets.loc[fdp, 'prad'] / radratio
                comp.at[res, 'radj_err1'] = comp.at[res, 'rade_err1'] / radratio
                comp.at[res, 'radj_err2'] = comp.at[res, 'rade_err2'] / radratio

        # make sure all candidate KOIs have the new parameters
        for index, ican in dfkoi[koican].iterrows():
            res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                           (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                           (np.abs((comp['period'] - ican['period']) /
                                   ican['period']) < 0.01))
            res = res[0]
            assert len(res) == 0

            fdp = np.where(koiplanets['id_planet'] == ican['name'])[0]
            fds = np.where(koistars['id_starname'] == f'kic{ican["IC"]}')[0]
            if len(fdp) != 1 or len(fds) != 1:
                continue
            else:
                fdp = fdp[0]
                fds = fds[0]
                dfkoi.at[index, 'st_mass'] = koistars.loc[fds, 'iso_mass']
                dfkoi.at[index, 'st_rad'] = koistars.loc[fds, 'iso_rad']
                dfkoi.at[index, 'st_teff'] = koistars.loc[fds, 'iso_teff']
                l10 = np.log10(koistars.loc[fds, 'iso_lum'])
                dfkoi.at[index, 'st_log_lum'] = l10
                dfkoi.at[index, 'distance_pc'] = koistars.loc[fds, 'iso_dis']

                dfkoi.at[index, 'semi_au'] = koiplanets.loc[fdp, 'sma']
                dfkoi.at[index, 'insol'] = koiplanets.loc[fdp, 'insol']
                dfkoi.at[index, 'rade'] = koiplanets.loc[fdp, 'prad']
                dfkoi.at[index, 'rade_err1'] = koiplanets.loc[fdp, 'prad_err1']
                dfkoi.at[index, 'rade_err2'] = koiplanets.loc[fdp, 'prad_err2']
                dfkoi.at[index, 'radj'] = koiplanets.loc[fdp, 'prad'] / radratio
                dfkoi.at[index, 'radj_err1'] = (dfkoi.at[index, 'rade_err1'] /
                                                radratio)
                dfkoi.at[index, 'radj_err2'] = (dfkoi.at[index, 'rade_err2'] /
                                                radratio)
        """
        # by definition these are missing from the Berger sample, but I'm
        # keeping this cross-match in case we need it later.
        missing = ['KOI-142 c', 'Kepler-78 b', 'Kepler-16 b', 'Kepler-34 b',
                   'Kepler-35 b', 'Kepler-38 b', 'Kepler-47 b', 'Kepler-47 c',
                   'PH1 b', 'KOI-55 b', 'KOI-55 c', 'Kepler-1647 b',
                   'Kepler-413 b', '2MASS J19383260+4603591 b', 'Kepler-453 b',
                   'Kepler-1654 b', 'Kepler-1661 b', 'Kepler-448 c',
                   'Kepler-88 d', 'Kepler-47 d', 'HAT-P-11 c', 'Kepler-90 i',
                   'Kepler-1708 b', 'Kepler-451 c', 'Kepler-451 d', 'KOI-984 c']
        fillkics = [5446285, 8435766, 12644769, 8572936, 9837578, 6762829,
                    10020423, 10020423, 4862625, 5807616, 5807616, 5473556,
                    12351927, 9472174, 9632895, 8410697, 6504534, 5812701,
                    5446285, 10020423, 10748390, 11442793, 7906827, 9472174,
                    9472174, 1161345]
        """

    koicon = dfkoi['disposition'] == 'Confirmed'
    koican = dfkoi['disposition'] == 'Candidate'
    # candidates where the planet is bigger than the star shouldn't exist
    badfit = dfkoi['koi_ror'] > 1
    assert not (badfit & koicon).any()
    # make sure things that have radii also have depths and vice versa
    cons = ((~np.isfinite(dfkoi['tran_depth_ppm'])) &
            np.isfinite(dfkoi['rade'])).sum()
    assert cons == 0

    # all KOIs transit
    dfkoi['flag_tran'] = True

    # all KOIs were observed by Kepler but not K2
    dfkoi['flag_kepler'] = True
    dfkoi['flag_k2'] = False

    # these have not been confirmed
    dfkoi['year_confirmed'] = np.nan

    # all discovered via transit
    dfkoi['discoverymethod'] = 'Transit'

    # no masses
    dfkoi['masse'] = np.nan
    dfkoi['masse_err1'] = np.nan
    dfkoi['masse_err2'] = np.nan
    dfkoi['massj'] = np.nan
    dfkoi['massj_err1'] = np.nan
    dfkoi['massj_err2'] = np.nan

    # no eccentricity
    dfkoi['eccen'] = np.nan

    # do all the tests to make sure things are working, only focusing on
    # candidates since that's all we're adding to output. FPs can be weird.
    canonly = dfkoi[koican]

    # everything has a facility name
    assert (np.array([len(xi) for xi in canonly['facility']]) > 0).all()
    assert (canonly['disposition'] == 'Candidate').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in canonly['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in canonly['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in canonly['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert (canonly['IC'] > 0).all()
    assert (canonly['discoverymethod'] == 'Transit').all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(canonly['distance_pc']) |
            (canonly['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(canonly['st_rad']) | (canonly['st_rad'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_mass']) |
            (canonly['st_mass'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_teff']) |
            (canonly['st_teff'] > 3000)).all()
    assert (~np.isfinite(canonly['st_log_lum']) |
            ((canonly['st_log_lum'] > -3) & (canonly['st_log_lum'] < 5))).all()
    assert (~np.isfinite(canonly['Kmag']) | (canonly['Kmag'] > 0)).all()
    assert (~np.isfinite(canonly['Jmag']) | (canonly['Jmag'] > 0)).all()

    # RA and Dec are both valid
    assert ((canonly['ra'] >= 0) & (canonly['ra'] <= 360.)).all()
    assert ((canonly['dec'] >= -90) & (canonly['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert (canonly['period'] > 0).all()
    assert (~np.isfinite(canonly['semi_au']) | (canonly['semi_au'] > 0)).all()
    assert (~np.isfinite(canonly['insol']) | (canonly['insol'] > 0)).all()
    assert (~np.isfinite(canonly['rade']) | (canonly['rade'] > 0)).all()
    assert (~np.isfinite(canonly['rade_err1']) |
            (canonly['rade_err1'] >= 0)).all()
    assert (~np.isfinite(canonly['rade_err2']) |
            (canonly['rade_err2'] <= 0)).all()
    assert (~np.isfinite(canonly['radj']) | (canonly['radj'] > 0)).all()
    assert (~np.isfinite(canonly['radj_err1']) |
            (canonly['radj_err1'] >= 0)).all()
    assert (~np.isfinite(canonly['radj_err2']) |
            (canonly['radj_err2'] <= 0)).all()
    assert (~np.isfinite(canonly['masse'])).all()
    assert (~np.isfinite(canonly['masse_err1'])).all()
    assert (~np.isfinite(canonly['masse_err2'])).all()
    assert (~np.isfinite(canonly['massj'])).all()
    assert (~np.isfinite(canonly['massj_err1'])).all()
    assert (~np.isfinite(canonly['massj_err2'])).all()
    assert (~np.isfinite(canonly['eccen'])).all()
    assert (~np.isfinite(canonly['tran_depth_ppm']) |
            (canonly['tran_depth_ppm'] >= 0)).all()
    assert (canonly['tran_dur_hr'] > 0).all()

    # Jup and Earth radii are either defined or not together
    assert np.allclose(np.isfinite(canonly['radj']),
                       np.isfinite(canonly['rade']))
    assert ((~np.isfinite(canonly['rade'])) |
            ((canonly['rade'] / canonly['radj'] > 0.99 * radratio) &
             (canonly['rade'] / canonly['radj'] < 1.01 * radratio))).all()
    assert np.allclose(np.isfinite(canonly['radj_err1']),
                       np.isfinite(canonly['rade_err1']))
    assert ((~np.isfinite(canonly['rade_err1'])) | (canonly['rade_err1'] == 0) |
            ((canonly['rade_err1'] / canonly['radj_err1'] > 0.99 * radratio) &
             (canonly['rade_err1'] / canonly['radj_err1'] <
              1.01 * radratio))).all()
    assert np.allclose(np.isfinite(canonly['radj_err2']),
                       np.isfinite(canonly['rade_err2']))
    assert ((~np.isfinite(canonly['rade_err2'])) | (canonly['rade_err2'] == 0) |
            ((canonly['rade_err2'] / canonly['radj_err2'] > 0.99 * radratio) &
             (canonly['rade_err2'] / canonly['radj_err2'] <
              1.01 * radratio))).all()

    # these flags at least have the right number of good values
    assert canonly['flag_kepler'].all()
    assert not canonly['flag_k2'].any()
    assert canonly['flag_tran'].all()

    # discovery and confirmation years make sense
    assert ((canonly['year_discovered'] >= 2011) &
            (canonly['year_discovered'] <= 2018)).all()
    assert not np.isfinite(canonly['year_confirmed']).any()

    # add the KOIs to our final composite table
    koiadd = dfkoi[cols][koican].copy()
    comp = pd.concat((comp, koiadd), verify_integrity=True, ignore_index=True)

    ################
    # K2 LIST PREP #
    ################

    print('Handling K2OIs.')
    # put these into our keywords
    renames = {'pl_rade': 'rade', 'pl_rade_err1': 'rade_err1',
               'pl_rade_err2': 'rade_err2', 'pl_orbsmax': 'semi_au',
               'pl_insol': 'insol', 'pl_orbper': 'period', 'pl_radj': 'radj',
               'pl_radj_err1': 'radj_err1', 'pl_radj_err2': 'radj_err2',
               'sy_kmag': 'Kmag', 'pl_name': 'name', 'pl_orbeccen': 'eccen',
               'pl_trandep': 'tran_depth_ppm', 'pl_trandur': 'tran_dur_hr',
               'sy_jmag': 'Jmag', 'disc_year': 'year_discovered',
               'pl_bmasse': 'masse', 'pl_bmasseerr1': 'masse_err1',
               'pl_bmasseerr2': 'masse_err2', 'pl_bmassj': 'massj',
               'pl_bmassjerr1': 'massj_err1', 'pl_bmassjerr2': 'massj_err2',
               'st_lum': 'st_log_lum'}
    dfk2.rename(columns=renames, inplace=True)

    # ignore the microlensing one in the K2 table. it's handled in the confirmed
    lens = dfk2['name'] == 'K2-2016-BLG-0005L b'
    assert lens.sum() == 2
    dfk2 = dfk2[~lens]
    dfk2 = dfk2.reset_index(drop=True)

    # upper/lower limits are given values at that limit, and we need to remove
    # them for now
    dfk2.loc[dfk2['pl_orbperlim'] != 0, 'period'] = np.nan
    dfk2.loc[dfk2['pl_radelim'] != 0,
             ['rade', 'rade_err1', 'rade_err2']] = np.nan
    dfk2.loc[dfk2['pl_radjlim'] != 0,
             ['radj', 'radj_err1', 'radj_err2']] = np.nan
    dfk2.loc[dfk2['st_radlim'] != 0, 'st_rad'] = np.nan
    dfk2.loc[dfk2['st_tefflim'] != 0, 'st_teff'] = np.nan
    dfk2.loc[dfk2['pl_ratdorlim'] != 0, 'pl_ratdor'] = np.nan
    dfk2.loc[dfcon['pl_trandeplim'] != 0, 'tran_depth_ppm'] = np.nan
    dfk2.loc[dfcon['pl_trandurlim'] != 0, 'tran_dur_hr'] = np.nan
    dfk2.loc[dfcon['pl_ratrorlim'] != 0, 'pl_ratror'] = np.nan

    # make sure we have both values defined
    noearth = (~np.isfinite(dfk2['rade']) & np.isfinite(dfk2['radj']))
    assert noearth.sum() == 0
    nojup = (np.isfinite(dfk2['rade']) & (~np.isfinite(dfk2['radj'])))
    assert nojup.sum() == 0

    # because earth and jup radii don't always agree, make them
    # uniform and treat Earth as truth
    dfk2['radj'] = dfk2['rade'] / radratio
    dfk2['radj_err1'] = dfk2['rade_err1'] / radratio
    dfk2['radj_err2'] = dfk2['rade_err2'] / radratio

    # make these not all caps
    dfk2['disposition'] = dfk2['disposition'].str.title()
    assert np.unique(dfk2['disposition']).size == 4

    # avoid a defrag warning without dealing with cleaning up the code
    dfk2 = dfk2.copy()
    # set the appropriate discovery facility for candidates
    dfk2['facility'] = 'K2'

    # make an int column of EPICs
    epics = []
    for iep in dfk2['epic_hostname']:
        epics.append(int(iep[4:]))
    epics = np.array(epics)
    dfk2['IC'] = epics
    dfk2['TIC'] = 0
    dfk2['distance_pc'] = 0.0

    # match EPICs to TICs and Gaia distances
    epics, tics, epdists = np.loadtxt(k2distfile, unpack=True)
    epics = epics.astype(int)
    tics = tics.astype(int)

    mtics = dfk2['IC'].values * 0
    for ii in np.arange(mtics.size):
        match = dfk2.loc[ii, 'IC'] == epics
        assert match.sum() == 1
        dfk2.loc[ii, 'TIC'] = tics[match]
        dfk2.loc[ii, 'distance_pc'] = epdists[match]

    # where do we want to point people to on clicking?
    u2 = 'https://exofop.ipac.caltech.edu/tess/target.php?id='
    dfk2['url'] = (u2 + dfk2['TIC'].astype(str))

    # check that we're including all K2 planets, but only counting them once
    k2con = dfk2['disposition'] == 'Confirmed'
    k2can = dfk2['disposition'] == 'Candidate'

    # all K2 confirmed planets are already in the confirmed planets table
    notfound = ~np.isin(dfk2['name'][k2con], comp['name'])
    assert notfound.sum() == 0

    # anything with a planet name in the K2 table but still a candidate hasn't
    # already shown up in the confirmed planets table
    hasname = ~dfk2['name'][k2can].isna()
    assert np.isin(dfk2['name'][k2can][hasname], comp['name']).sum() == 0

    uobjs = np.unique(dfk2['name'])
    bad = []
    for iobj in uobjs:
        vv = np.where(dfk2['name'] == iobj)[0]
        if np.unique(dfk2['disposition'][vv]).size != 1:
            bad.append(iobj)
        assert np.unique(dfk2['disposition'][vv]).size == 1

    # the Kruse sample got K2-19 and K2-24 periods wrong,
    # Crossfield got K2-22 wrong, Kruse/Mayo disagree on K2-189 by 2x
    # the rest are usually a NaN period
    # V1298 e is a refined period from a single transit in K2 to another with
    # TESS
    k2exclude = ['HD 3167 c', 'HD 3167 d', 'HD 3167 e', 'HIP 41378 d',
                 'HIP 41378 e', 'HIP 41378 f', 'K2-132 b', 'K2-189 b',
                 'K2-19 c', 'K2-22 b', 'K2-24 c', 'K2-290 b', 'K2-290 c',
                 'K2-97 b', 'TRAPPIST-1 b', 'TRAPPIST-1 c', 'TRAPPIST-1 d',
                 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g', 'TRAPPIST-1 h',
                 'V1298 Tau e', 'WASP-107 b', 'WASP-47 b',
                 'WASP-47 c', 'K2-79 b', 'K2-26 b']

    isexclude = np.zeros(len(k2exclude), dtype=bool)

    # make sure all confirmed K2 planets are in the confirmed table exactly once
    # and update their discovery year
    for index, icon in dfk2[k2con].iterrows():
        res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - icon['period']) /
                               icon['period']) < 0.01))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can ignore
            assert len(res) == 0
            assert icon['name'] in k2exclude
            isexclude[k2exclude.index(icon['name'])] = True
            res = np.where(comp['name'] == icon['name'])[0]
            assert len(res) == 1
        # update and sync the discovery year in both tables
        res = res[0]
        assert comp.at[res, 'name'] == icon['name']
        myr = min(comp.at[res, 'year_discovered'], icon['year_discovered'])
        comp.at[res, 'year_discovered'] = myr
        dfk2.at[index, 'year_discovered'] = myr

    assert isexclude.all()

    # these are confirmed planets that aren't listed as such, so match them up
    # and set them as confirmed
    # XXX: notify archive
    k2known = ['EPIC 212555594.02', 'EPIC 201357835.01']
    plname = ['K2-192 b', 'K2-245 b']
    reknown = np.zeros(len(k2known), dtype=bool)

    # make sure all candidate K2 planets aren't in the confirmed table
    for index, ican in dfk2[k2can].iterrows():
        res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - ican['period']) /
                               ican['period']) < 0.01))
        res = res[0]
        if len(res) != 0:
            assert ican['name'] in k2known
            pall = dfk2['name'] == ican['name']
            dfk2.loc[pall, 'disposition'] = 'Confirmed'
            dfk2.loc[pall, 'name'] = plname[k2known.index(ican['name'])]
            reknown[k2known.index(ican['name'])] = True

            # EPIC 201357835.01 is K2-245, but that has a different
            # EPIC: 201357643
            if ican['name'] == 'EPIC 201357835.01':
                dfk2.at[index, 'IC'] = 201357643
                dfk2.at[index, 'name'] = 'K2-245 b'
                dfk2.at[index, 'hostname'] = 'EPIC 201357643'
                dfk2.at[index, 'default_flag'] = 0
                u1 = 'https://exofop.ipac.caltech.edu/k2/edit_target.php?id='
                dfk2.at[index, 'url'] = (u1 + dfk2.at[index, 'hostname'][5:])

            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], ican['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dfk2.at[index, 'year_discovered'] = myr

    assert reknown.all()

    dfk2['flag_tran'] = dfk2['tran_flag'].values.astype(bool)

    # convert their depth in % to depth in ppm
    dfk2['tran_depth_ppm'] *= 1e4
    # we should have transit depths for these
    getdep = (dfk2['flag_tran'] & (~np.isfinite(dfk2['tran_depth_ppm'])) &
              np.isfinite(dfk2['rade']) & np.isfinite(dfk2['st_rad']))

    tranrat = dfk2['rade']**2 / (dfk2['st_rad'] * sunearth)**2
    dfk2.loc[getdep, 'tran_depth_ppm'] = tranrat[getdep] * 1e6

    # we should have a radius if they gave a depth
    getrad = (dfk2['flag_tran'] & np.isfinite(dfk2['tran_depth_ppm']) &
              (~np.isfinite(dfk2['rade'])) & np.isfinite(dfk2['st_rad']))
    tranrad = np.sqrt((dfk2['tran_depth_ppm']/1e6) * (dfk2['st_rad']**2))
    tranrad *= sunearth
    dfk2.loc[getrad, 'rade'] = tranrad[getrad]
    dfk2.loc[getrad, 'radj'] = tranrad[getrad] / radratio

    # fill in any missing luminosities with our own calculation
    tmplums = (dfk2['st_rad']**2) * ((dfk2['st_teff'] / 5772)**4)
    toadd = (~np.isfinite(dfk2['st_log_lum'])) & np.isfinite(tmplums)
    dfk2.loc[toadd, 'st_log_lum'] = np.log10(tmplums[toadd])

    # fill in any missing semi-major axes from Kepler's third law first
    tmpau = (((dfk2['period'] / 365.25)**2) * dfk2['st_mass'])**(1./3.)
    repau = (~np.isfinite(dfk2['semi_au'])) & np.isfinite(tmpau)
    dfk2.loc[repau, 'semi_au'] = tmpau[repau]

    # then fill in any missing semi-major axes with a/R* * R*
    # convert to AU; 1 AU = 215 Rsun
    tmpau2 = dfk2['pl_ratdor'] * dfk2['st_rad'] / (const.au/const.R_sun).value
    repau2 = (~np.isfinite(dfk2['semi_au'])) & np.isfinite(tmpau2)
    dfk2.loc[repau2, 'semi_au'] = tmpau2[repau2]

    # calculate insolations ourselves and fill in any missing that we can
    tmpinsol = (10.**dfk2['st_log_lum']) * (dfk2['semi_au']**-2)
    repinsol = (~np.isfinite(dfk2['insol'])) & np.isfinite(tmpinsol)
    dfk2.loc[repinsol, 'insol'] = tmpinsol[repinsol]

    # no K2 candidate observed by Kepler prime but all by K2
    dfk2['flag_kepler'] = False
    dfk2['flag_k2'] = True

    # these have not been confirmed
    dfk2['year_confirmed'] = np.nan

    k2can = (dfk2['disposition'] == 'Candidate') & (dfk2['default_flag'] == 1)

    assert k2can.sum() == np.unique(dfk2[dfk2['disposition'] ==
                                    'Candidate']['name']).size

    # go through the ones we're going to add to the table and fill in any
    # missing values if possible from other entries in the table
    ichk = ['period', 'rade', 'rade_err1', 'rade_err2', 'radj', 'radj_err1',
            'radj_err2', 'st_rad', 'st_teff', 'st_log_lum', 'st_mass', 'insol',
            'semi_au', 'tran_depth_ppm', 'eccen']

    for index, ican in dfk2[k2can].iterrows():
        srch = np.where((dfk2['name'] == ican['name']) &
                        (dfk2['default_flag'] == 0))[0]
        if len(srch) == 0:
            continue
        for icol in ichk:
            if ((not np.isfinite(dfk2.at[index, icol])) and
                    np.isfinite(dfk2[icol][srch]).any()):
                dfk2.at[index, icol] = np.nanmedian(dfk2[icol][srch])

    # do all the tests to make sure things are working, only focusing on
    # candidates since that's all we're adding to output. FPs can be weird.
    canonly = dfk2[k2can]

    # everything has a facility name
    assert (np.array([len(xi) for xi in canonly['facility']]) > 0).all()
    assert (canonly['disposition'] == 'Candidate').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in canonly['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in canonly['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in canonly['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert (canonly['IC'] > 0).all()
    assert (canonly['discoverymethod'] == 'Transit').all()

    assert (~np.isfinite(canonly['eccen']) |
            ((canonly['eccen'] >= 0) & (canonly['eccen'] < 1))).all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(canonly['distance_pc']) |
            (canonly['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(canonly['st_rad']) | (canonly['st_rad'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_mass']) |
            (canonly['st_mass'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_teff']) |
            (canonly['st_teff'] > 2900)).all()
    assert (~np.isfinite(canonly['st_log_lum']) |
            ((canonly['st_log_lum'] > -3) & (canonly['st_log_lum'] < 7))).all()
    assert (~np.isfinite(canonly['Kmag']) | (canonly['Kmag'] > 3)).all()
    assert (~np.isfinite(canonly['Jmag']) | (canonly['Jmag'] > 4)).all()

    # RA and Dec are both valid
    assert ((canonly['ra'] >= 0) & (canonly['ra'] <= 360.)).all()
    assert ((canonly['dec'] >= -90) & (canonly['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert ((~np.isfinite(canonly['period'])) | (canonly['period'] > 0)).all()
    assert (~np.isfinite(canonly['semi_au']) | (canonly['semi_au'] > 0)).all()
    assert (~np.isfinite(canonly['insol']) | (canonly['insol'] > 0)).all()
    assert (~np.isfinite(canonly['rade']) | (canonly['rade'] > 0)).all()
    assert (~np.isfinite(canonly['rade_err1']) |
            (canonly['rade_err1'] >= 0)).all()
    assert (~np.isfinite(canonly['rade_err2']) |
            (canonly['rade_err2'] <= 0)).all()
    assert (~np.isfinite(canonly['radj']) | (canonly['radj'] > 0)).all()
    assert (~np.isfinite(canonly['radj_err1']) |
            (canonly['radj_err1'] >= 0)).all()
    assert (~np.isfinite(canonly['radj_err2']) |
            (canonly['radj_err2'] <= 0)).all()
    assert (~np.isfinite(canonly['masse'])).all()
    assert (~np.isfinite(canonly['masse_err1'])).all()
    assert (~np.isfinite(canonly['masse_err2'])).all()
    assert (~np.isfinite(canonly['massj'])).all()
    assert (~np.isfinite(canonly['massj_err1'])).all()
    assert (~np.isfinite(canonly['massj_err2'])).all()
    assert (~np.isfinite(canonly['tran_depth_ppm']) |
            (canonly['tran_depth_ppm'] > 0)).all()
    assert (~np.isfinite(canonly['tran_dur_hr']) |
            (canonly['tran_dur_hr'] >= 0)).all()

    # Jup and Earth radii are either defined or not together
    assert np.allclose(np.isfinite(canonly['radj']),
                       np.isfinite(canonly['rade']))
    assert ((~np.isfinite(canonly['rade'])) |
            ((canonly['rade'] / canonly['radj'] > 0.99 * radratio) &
             (canonly['rade'] / canonly['radj'] < 1.01 * radratio))).all()
    assert np.allclose(np.isfinite(canonly['radj_err1']),
                       np.isfinite(canonly['rade_err1']))
    assert ((~np.isfinite(canonly['rade_err1'])) |
            ((canonly['rade_err1'] / canonly['radj_err1'] > 0.99 * radratio) &
             (canonly['rade_err1'] / canonly['radj_err1'] <
              1.01 * radratio))).all()
    assert np.allclose(np.isfinite(canonly['radj_err2']),
                       np.isfinite(canonly['rade_err2']))
    assert ((~np.isfinite(canonly['rade_err2'])) |
            ((canonly['rade_err2'] / canonly['radj_err2'] > 0.99 * radratio) &
             (canonly['rade_err2'] / canonly['radj_err2'] <
              1.01 * radratio))).all()

    # these flags at least have the right number of good values
    assert not canonly['flag_kepler'].any()
    assert canonly['flag_k2'].all()
    assert canonly['flag_tran'].all()

    # discovery and confirmation years make sense
    assert (canonly['year_discovered'] >= 2015).all()
    assert not np.isfinite(canonly['year_confirmed']).any()

    # add the K2 candidates to our final composite table
    k2add = dfk2[cols][k2can].copy()
    comp = pd.concat((comp, k2add), verify_integrity=True, ignore_index=True)

    #################
    # TOI LIST PREP #
    #################

    print('Handling TOIs.')
    # get easier to reference names for things in the ExoFOP listing
    renames = {'TFOPWG Disposition': 'disposition', 'TIC ID': 'IC',
               'Period (days)': 'period', 'Planet Radius (R_Earth)': 'rade',
               'Planet Radius (R_Earth) err': 'rade_err1',
               'Stellar Radius (R_Sun)': 'st_rad',
               'Stellar Eff Temp (K)': 'st_teff',
               'Stellar Mass (M_Sun)': 'st_mass',
               'Stellar Distance (pc)': 'distance_pc',
               'Planet Insolation (Earth Flux)': 'insol',
               'Depth (ppm)': 'tran_depth_ppm',
               'Duration (hours)': 'tran_dur_hr'}
    dftoi.rename(columns=renames, inplace=True)

    # set these to strings we'd want to show in a figure
    dftoi['name'] = 'TOI-' + dftoi['TOI'].astype(str)
    dftoi['hostname'] = 'TIC ' + dftoi['IC'].astype(str)

    # download TIC info for any new entries
    if (fulltic is None) or (not np.isin(dftoi['IC'], fulltic['ID']).all()):
        from astroquery.mast import Catalogs
        for index, irow in dftoi.iterrows():
            if (fulltic is None) or (irow['IC'] not in fulltic['ID'].values):
                print('Getting TIC info', index, dftoi['IC'].size - 1)
                cat = Catalogs.query_criteria(catalog='tic', ID=irow['IC'])
                assert len(cat) == 1 and int(cat['ID'][0]) == irow['IC']
                head, istr = cat.to_pandas().to_csv().split()
                if not os.path.exists(ticparams):
                    with open(ticparams, 'w') as off:
                        off.write(head + '\n')
                with open(ticparams, 'a') as off:
                    off.write(istr + '\n')
                fulltic = pd.read_csv(ticparams)

    assert np.isin(dftoi['IC'], fulltic['ID']).all()

    # by default, we assume new candidates aren't in Kepler/K2 data
    dftoi['flag_kepler'] = False
    dftoi['flag_k2'] = False

    # orbital periods are either NaN or > 0 days
    noper = dftoi['period'] == 0
    dftoi.loc[noper, 'period'] = np.nan
    assert (~np.isfinite(dftoi['period']) | (dftoi['period'] > 0)).all()

    # things that don't have a disposition get PC
    dftoi['disposition'] = dftoi['disposition'].replace(np.nan, 'PC')
    # change this to the status we want to report
    dftoi['disposition'] = dftoi['disposition'].replace('PC', 'Candidate')
    dftoi['disposition'] = dftoi['disposition'].replace('KP', 'Confirmed')
    dftoi['disposition'] = dftoi['disposition'].replace('CP', 'Confirmed')
    dftoi['disposition'] = dftoi['disposition'].replace('APC', 'Candidate')
    dftoi['disposition'] = dftoi['disposition'].replace('FA', 'False Positive')
    dftoi['disposition'] = dftoi['disposition'].replace('FP', 'False Positive')
    assert np.unique(dftoi['disposition']).size == 3

    # make these useful degrees like all the other catalogs
    dftoi['ra'] = Angle(dftoi['RA'].values, unit='hourangle').degree
    dftoi['dec'] = Angle(dftoi['Dec'].values, unit='degree').degree

    # assume symmetrical errors as they seem to do
    dftoi['rade_err2'] = -1 * dftoi['rade_err1']
    # give TOIs units of Jupiter radii
    dftoi['radj'] = dftoi['rade'] / radratio
    dftoi['radj_err1'] = dftoi['rade_err1'] / radratio
    dftoi['radj_err2'] = dftoi['rade_err2'] / radratio

    # set the appropriate discovery facility for candidates
    dftoi['facility'] = 'TESS'

    # where do we want to point people to on clicking?
    u2 = 'https://exofop.ipac.caltech.edu/tess/target.php?id='
    dftoi['url'] = (u2 + dftoi['IC'].astype(str))

    # the year the TOI was found
    yrs = []
    for ival in dftoi['Date TOI Alerted (UTC)']:
        yrs.append(int(ival[:4]))
    dftoi['year_discovered'] = yrs
    assert (np.array(yrs) > 2017).all()

    # these are brown dwarfs and not real planets (TOI-4571 is a demoted KOI)
    bds = ['TOI-239.01', 'TOI-1779.01', 'TOI-148.01', 'TOI-503.01',
           'TOI-569.01', 'TOI-629.01', 'TOI-1406.01', 'TOI-1417.01',
           'TOI-2119.01', 'TOI-1278.01', 'TOI-2543.01', 'TOI-5081.01',
           'TOI-5090.01', 'TOI-4571.01', 'TOI-6970.01', 'TOI-4454.01',
           'TOI-2336.01', 'TOI-2844.01', 'TOI-3577.01', 'TOI-3755.01',
           'TOI-4462.01', 'TOI-4737.01', 'TOI-4759.01', 'TOI-5240.01',
           'TOI-5467.01', 'TOI-5882.01']
    for ibd in bds:
        bd = np.where(dftoi['name'] == ibd)[0][0]
        assert dftoi.loc[bd, 'disposition'] == 'Confirmed'
        dftoi.loc[bd, 'disposition'] = 'False Positive'

    # the TOI list from ExoFOP isn't always kept synced with the confirmed
    # planets table, so do some shifting of categories here.
    # match planets between tables by RA/Dec/Period
    toicon = dftoi['disposition'] == 'Confirmed'
    toican = dftoi['disposition'] == 'Candidate'

    # any supposedly confirmed TOIs that aren't in the table get demoted back
    # to candidate

    # these we have an explanation for and know they're properly in the
    # confirmed table. 2011, 2221 are single transits, so no period matching.
    # 351 TESS got the period wrong by 2x. TOI-561 was 2 transits of different
    # planets.
    ignores = ['TOI-2011.03','TOI-2221.01', 'TOI-4581.02', 'TOI-5980.01',
               'TOI-2319.01', 'TOI-216.02', 'TOI-6083.01', 'TOI-561.04',
               'TOI-6087.01', 'TOI-4862.01', 'TOI-6962.01',
               'TOI-2447.01', 'TOI-2529.01', 'TOI-6883.01', 'TOI-6984.01',
               'TOI-1408.02', 'TOI-7068.01', 'TOI-2010.01']
    conname = ['HD 136352 d', 'AU Mic b', 'KOI-94 e', 'Kepler-37 d',
               'HD 152843 c', 'TOI-216.02', 'Kepler-858 b', 'TOI-561 e',
               'Kepler-134 b', 'NGTS-30 b', 'K2-79 b',
               'TOI-2447 b', 'TOI-2529 b', 'TIC 393818343 b', 'K2-26 b',
               'TOI-1408 c', 'Kepler-1514 b', 'TOI-2010 b']
    # we know what these are, and they have paper trails of submitted papers
    # though some were submitted way back in 2014 and still in limbo
    # some are newly submitted and waiting to be accepted but are
    # prematurely marked confirmed on ExoFOP
    waiting = ['TOI-126.01', 'TOI-143.01', 'TOI-295.01', 'TOI-626.01',
               'TOI-657.01', 'TOI-834.01', 'TOI-840.01', 'TOI-857.01',
               'TOI-1071.01', 'TOI-1603.01', 'TOI-2330.01', 'TOI-261.02',
               'TOI-682.01', 'TOI-1054.01', 'TOI-1203.01', 'TOI-1230.01',
               'TOI-1239.01', 'TOI-1774.01', 'TOI-263.01', 'TOI-3422.01',
               'TOI-3666.01', 'TOI-5153.01', 'TOI-5812.01', 'TOI-1260.03',
               'TOI-6101.01', 'TOI-6170.01', 'TOI-5678.01', 'TOI-603.01',
               'TOI-1605.01', 'TOI-1828.01', 'TOI-1885.01',
               'TOI-2215.01', 'TOI-7041.01', 'TOI-512.01']
    earlycps = []

    stillbad = np.zeros(len(ignores), dtype=bool)
    stillwaiting = np.zeros(len(waiting), dtype=bool)
    for index, icon in dftoi[toicon].iterrows():
        res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - icon['period']) /
                               comp['period']) < 0.01))
        res = res[0]
        if len(res) == 0:
            # keep these as confirmed, they just failed matching
            if icon['name'] in ignores:
                stillbad[ignores.index(icon['name'])] = True
                res = np.where(comp['name'] ==
                               conname[ignores.index(icon['name'])])[0]
            # these haven't made the confirmed table yet, so they're
            # officially still candidates
            else:
                if icon['name'] not in waiting:
                    earlycps.append(icon['name'])
                else:
                    stillwaiting[waiting.index(icon['name'])] = True
                    dftoi.at[index, 'disposition'] = 'Candidate'

            # try looking for just same location on the sky and make sure
            # nothing new comes up
            if ~np.isfinite(icon['period']):
                res2 = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                                (np.abs(comp['dec'] - icon['dec']) < 1. / 60))
                res2 = res2[0]
                if len(res2) > 0:
                    assert icon['name'] in ignores
        assert len(res) < 2
        if len(res) == 1:
            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], icon['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dftoi.at[index, 'year_discovered'] = myr

    assert stillbad.all()
    assert stillwaiting.all()
    assert len(earlycps) == 0

    # these are now confirmed and need to be updated as such
    tobeconf = ['TOI-1194.01', 'TOI-1420.01', 'TOI-5344.01', 'TOI-2134.01',
                'TOI-4481.01', 'TOI-4641.01', 'TOI-2266.01', 'TOI-5747.01',
                'TOI-1347.01', 'TOI-1347.02', 'TOI-771.01', 'TOI-871.01',
                'TOI-1467.01', 'TOI-1739.01', 'TOI-1778.01', 'TOI-1994.01',
                'TOI-2068.01', 'TOI-4559.01', 'TOI-5799.01', 'TOI-128.01',
                'TOI-260.01', 'TOI-261.01', 'TOI-266.01', 'TOI-741.01',
                'TOI-286.01', 'TOI-286.02', 'TOI-406.01', 'TOI-486.01',
                'TOI-554.01', 'TOI-554.02', 'TOI-654.01', 'TOI-663.01',
                'TOI-663.02', 'TOI-663.03', 'TOI-669.01', 'TOI-757.01',
                'TOI-762.01', 'TOI-782.01', 'TOI-815.01', 'TOI-880.02',
                'TOI-907.01', 'TOI-1174.01', 'TOI-1180.01', 'TOI-1184.01',
                'TOI-1224.01', 'TOI-1224.02', 'TOI-1244.01', 'TOI-1247.01',
                'TOI-1248.01', 'TOI-1249.01', 'TOI-1269.01', 'TOI-1279.01',
                'TOI-1410.01', 'TOI-1437.01', 'TOI-1443.01', 'TOI-1448.01',
                'TOI-1450.01', 'TOI-1451.01', 'TOI-1472.01', 'TOI-1473.01',
                'TOI-1669.01', 'TOI-1683.01', 'TOI-1691.01', 'TOI-1723.01',
                'TOI-1742.01', 'TOI-1753.01', 'TOI-1758.01', 'TOI-1775.01',
                'TOI-1776.01', 'TOI-1794.01', 'TOI-1798.01', 'TOI-1798.02',
                'TOI-1799.01', 'TOI-1806.01', 'TOI-1823.01', 'TOI-1824.01',
                'TOI-1836.02', 'TOI-1855.01', 'TOI-2015.01', 'TOI-2088.01',
                'TOI-2107.01', 'TOI-2120.01', 'TOI-2128.01', 'TOI-2134.02',
                'TOI-2368.01', 'TOI-2374.01', 'TOI-2379.01', 'TOI-2384.01',
                'TOI-2714.01', 'TOI-2981.01', 'TOI-3071.01', 'TOI-3218.01',
                'TOI-3261.01', 'TOI-3321.01', 'TOI-3353.01', 'TOI-3894.01',
                'TOI-3919.01', 'TOI-4153.01', 'TOI-4320.01', 'TOI-4379.01',
                'TOI-4443.01', 'TOI-4495.01', 'TOI-4527.01',
                'TOI-4602.01', 'TOI-4633.01', 'TOI-4914.01', 'TOI-5082.01',
                'TOI-5218.01', 'TOI-5232.01', 'TOI-5301.01', 'TOI-5388.01',
                'TOI-5616.01', 'TOI-5634.01', 'TOI-5720.01', 'TOI-6008.01',
                'TOI-6029.01', 'TOI-6034.01', 'TOI-6086.01', 'TOI-6255.01',
                'TOI-6982.01', 'TOI-2420.01', 'TOI-2485.01',
                'TOI-1883.01', 'TOI-2274.01', 'TOI-2768.01', 'TOI-6002.01',
                'TOI-1301.01', 'TOI-1630.01', 'TOI-1659.01', 'TOI-1716.01',
                'TOI-1744.01', 'TOI-1768.01', 'TOI-1772.01', 'TOI-1777.01',
                'TOI-1782.01', 'TOI-1782.02', 'TOI-2211.01', 'TOI-4638.01',
                'TOI-5726.01', 'TOI-6442.01',
                'TOI-6963.01', 'TOI-4504.01', 'TOI-4504.02',
                'TOI-1295.01', 'TOI-2580.01', 'TOI-5108.01', 'TOI-5786.01',
                'TOI-6016.01', 'TOI-6130.01', 'TOI-6276.01', 'TOI-6276.02',
                'TOI-2328.01', 'TOI-2537.01', 'TOI-3837.01', 'TOI-5027.01',
                'TOI-5110.01', 'TOI-6628.01',
                # KOIs
                'TOI-4444.01', 'TOI-4484.01', 'TOI-4588.01', 'TOI-1241.01',
                # K2 candidates
                'TOI-2410.01', 'TOI-2425.01', 'TOI-2455.01', 'TOI-2639.01',
                'TOI-4540.01', 'TOI-4549.01', 'TOI-4608.01', 'TOI-4611.01',
                'TOI-4615.01', 'TOI-4619.01', 'TOI-5073.01',
                'TOI-5102.01', 'TOI-5103.01', 'TOI-5105.01', 'TOI-5116.01',
                'TOI-5137.01', 'TOI-5140.01', 'TOI-5154.01', 'TOI-5158.01',
                'TOI-5161.01', 'TOI-5165.01', 'TOI-5167.01', 'TOI-5171.01',
                'TOI-5175.01', 'TOI-5176.01', 'TOI-5115.01', 'TOI-5480.01',
                'TOI-5522.01', 'TOI-5538.01', 'TOI-5539.01', 'TOI-5544.01',
                'TOI-5545.01', 'TOI-5561.01', 'TOI-6831.01', 'TOI-6832.01',
                'TOI-6840.01', 'TOI-6906.01', 'TOI-6966.01']
    tobeadded = []
    tbc = np.zeros(len(tobeconf), dtype=bool)
    # single transits that should be set as confirmed
    nopermatch = ['TOI-6697.02', 'TOI-1835.02']
    confmatch = ['HD 114082 b', 'HD 110067 d']
    singconf = np.zeros(len(nopermatch), dtype=bool)
    singcands = ['TOI-5523.01', 'TOI-1301.02', 'TOI-1772.02']
    singc = np.zeros(len(singcands), dtype=bool)

    # any candidates in the confirmed table get set as such
    for index, ican in dftoi[toican].iterrows():
        res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - ican['period']) /
                               comp['period']) < 0.01))
        res = res[0]
        # period and location match, so mark this as confirmed
        if len(res) == 1:
            dftoi.at[index, 'disposition'] = 'Confirmed'
            if ican['name'] not in tobeconf:
                tobeadded.append(ican['name'])
            else:
                tbc[tobeconf.index(ican['name'])] = True
            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], ican['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dftoi.at[index, 'year_discovered'] = myr
        else:
            assert len(res) == 0
            # try looking for just same location on the sky
            if ~np.isfinite(ican['period']):
                res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                               (np.abs(comp['dec'] - ican['dec']) < 1. / 60))[0]
                if len(res) > 0:
                    # keep it as a single transit candidate
                    if ican['name'] in singcands:
                        singc[singcands.index(ican['name'])] = True
                        continue
                    assert ican['name'] in nopermatch
                    res = np.where(comp['name'] ==
                                   confmatch[nopermatch.index(ican['name'])])[0]
                    assert len(res) == 1
                    # these are single transits of confirmed planets
                    dftoi.at[index, 'disposition'] = 'Confirmed'
                    singconf[nopermatch.index(ican['name'])] = True
                    # update and sync the discovery year in both tables
                    res = res[0]
                    myr = min(comp.at[res, 'year_discovered'],
                              ican['year_discovered'])
                    comp.at[res, 'year_discovered'] = myr
                    dftoi.at[index, 'year_discovered'] = myr

    assert tbc.all()
    assert singconf.all()
    assert singc.all()
    assert len(tobeadded) == 0

    newcan = dftoi['disposition'] == 'Candidate'
    newcon = dftoi['disposition'] == 'Confirmed'

    assert newcan.sum() == (toican.sum() + len(waiting) - len(tobeconf) -
                            len(nopermatch))
    assert newcon.sum() == (toicon.sum() + len(tobeconf) + len(nopermatch) -
                            len(waiting))

    # replace everything with new TIC parameters because the TOI parameters
    # rely on the TESS Project stellar parameters, which can sometimes be
    # a bit outdated. almost always doesn't make much difference though.
    dftoi[['Jmag', 'Kmag', 'st_log_lum']] = np.nan
    for index, itoi in dftoi.iterrows():
        mt = np.where(itoi['IC'] == fulltic['ID'])[0][0]
        # update to the TIC Teff, since the TESS Project ones seem outdated
        if np.isfinite(fulltic['Teff'][mt]):
            dftoi.at[index, 'st_teff'] = fulltic['Teff'][mt]
        if np.isfinite(fulltic['mass'][mt]):
            prev = dftoi.at[index, 'st_mass']
            # seems like the masses are always pulled straight from TIC 8.2
            if (not np.allclose(prev, fulltic['mass'][mt]) and np.isfinite(prev)
                    and itoi['disposition'] != 'False Positive'):
                raise Exception('bad m')
        if np.isfinite(fulltic['lum'][mt]) and fulltic['lum'][mt] > 0:
            dftoi.at[index, 'st_log_lum'] = np.log10(fulltic['lum'][mt])
        if np.isfinite(fulltic['rad'][mt]):
            prev = dftoi.at[index, 'st_rad']
            dftoi.at[index, 'st_rad'] = fulltic['rad'][mt]
            if np.isfinite(prev):
                dftoi.at[index, 'rade'] *= fulltic['rad'][mt]/prev
                dftoi.at[index, 'rade_err1'] *= fulltic['rad'][mt] / prev
                dftoi.at[index, 'rade_err2'] *= fulltic['rad'][mt] / prev
                dftoi.at[index, 'radj'] *= fulltic['rad'][mt]/prev
                dftoi.at[index, 'radj_err1'] *= fulltic['rad'][mt] / prev
                dftoi.at[index, 'radj_err2'] *= fulltic['rad'][mt] / prev
        dftoi.at[index, 'Kmag'] = fulltic['Kmag'][mt]
        dftoi.at[index, 'Jmag'] = fulltic['Jmag'][mt]

    # fill in missing luminosities with our own calculation
    tmplums = (dftoi['st_rad'] ** 2) * ((dftoi['st_teff'] / 5772) ** 4)
    tofill = ~np.isfinite(dftoi['st_log_lum'])
    dftoi.loc[tofill, 'st_log_lum'] = np.log10(tmplums[tofill])

    # get the semi-major axes and insolations
    iau = (((dftoi['period'] / 365.25)**2) * dftoi['st_mass'])**(1./3.)
    iinsol = (10.**dftoi['st_log_lum']) * (iau**-2)
    igd = np.isfinite(iinsol)
    dftoi['semi_au'] = iau
    dftoi.loc[igd, 'insol'] = iinsol[igd]

    # all TESS candidates transit
    dftoi['flag_tran'] = True

    # we should have transit depths for these (all TOIs so far have had
    # depth listed so this doesn't do anything)
    getdep = (dftoi['flag_tran'] & (~np.isfinite(dftoi['tran_depth_ppm'])) &
              np.isfinite(dftoi['rade']) & np.isfinite(dftoi['st_rad']))
    assert getdep.sum() == 0
    # tranrat = dftoi['rade']**2 / (dftoi['st_rad'] * sunearth)**2
    # dftoi.loc[getdep, 'tran_depth_ppm'] = tranrat[getdep] * 1e6

    # we should have a radius if they gave a depth
    getrad = (dftoi['flag_tran'] & np.isfinite(dftoi['tran_depth_ppm']) &
              (~np.isfinite(dftoi['rade'])) & np.isfinite(dftoi['st_rad']))
    tranrad = np.sqrt((dftoi['tran_depth_ppm']/1e6) * (dftoi['st_rad']**2))
    tranrad *= sunearth
    dftoi.loc[getrad, 'rade'] = tranrad[getrad]
    dftoi.loc[getrad, 'radj'] = tranrad[getrad] / radratio

    # these have not been confirmed
    dftoi['year_confirmed'] = np.nan

    # no masses
    dftoi['masse'] = np.nan
    dftoi['massj'] = np.nan
    dftoi['masse_err1'] = np.nan
    dftoi['massj_err1'] = np.nan
    dftoi['masse_err2'] = np.nan
    dftoi['massj_err2'] = np.nan

    dftoi['eccen'] = np.nan

    # all found via transit
    dftoi['discoverymethod'] = 'Transit'

    # do all the tests to make sure things are working, only focusing on
    # candidates since that's all we're adding to output. FPs can be weird.
    canonly = dftoi[newcan]

    # everything has a facility name
    assert (np.array([len(xi) for xi in canonly['facility']]) > 0).all()
    assert (canonly['disposition'] == 'Candidate').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in canonly['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in canonly['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in canonly['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert (canonly['IC'] > 0).all()
    assert (canonly['discoverymethod'] == 'Transit').all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(canonly['distance_pc']) |
            (canonly['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(canonly['st_rad']) | (canonly['st_rad'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_mass']) |
            (canonly['st_mass'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_teff']) |
            (canonly['st_teff'] > 2700)).all()
    assert (~np.isfinite(canonly['st_log_lum']) |
            ((canonly['st_log_lum'] > -3.1) & (canonly['st_log_lum'] < 7))).all()
    assert (~np.isfinite(canonly['Kmag']) | (canonly['Kmag'] > 0)).all()
    assert (~np.isfinite(canonly['Jmag']) | (canonly['Jmag'] > 0)).all()

    # RA and Dec are both valid
    assert ((canonly['ra'] >= 0) & (canonly['ra'] <= 360.)).all()
    assert ((canonly['dec'] >= -90) & (canonly['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert ((~np.isfinite(canonly['period'])) | (canonly['period'] > 0)).all()
    assert (~np.isfinite(canonly['semi_au']) | (canonly['semi_au'] > 0)).all()
    assert (~np.isfinite(canonly['insol']) | (canonly['insol'] > 0)).all()
    assert (~np.isfinite(canonly['rade']) | (canonly['rade'] > 0)).all()
    assert (~np.isfinite(canonly['rade_err1']) |
            (canonly['rade_err1'] >= 0)).all()
    assert (~np.isfinite(canonly['rade_err2']) |
            (canonly['rade_err2'] <= 0)).all()
    assert (~np.isfinite(canonly['radj']) | (canonly['radj'] > 0)).all()
    assert (~np.isfinite(canonly['radj_err1']) |
            (canonly['radj_err1'] >= 0)).all()
    assert (~np.isfinite(canonly['radj_err2']) |
            (canonly['radj_err2'] <= 0)).all()
    assert (~np.isfinite(canonly['masse'])).all()
    assert (~np.isfinite(canonly['massj'])).all()
    assert (~np.isfinite(canonly['eccen'])).all()
    assert (~np.isfinite(canonly['tran_depth_ppm']) |
            (canonly['tran_depth_ppm'] > 0)).all()
    assert (canonly['tran_dur_hr'] > 0).all()

    # Jup and Earth radii are either defined or not together
    assert np.allclose(np.isfinite(canonly['radj']),
                       np.isfinite(canonly['rade']))
    assert ((~np.isfinite(canonly['rade'])) |
            ((canonly['rade'] / canonly['radj'] > 0.99 * radratio) &
             (canonly['rade'] / canonly['radj'] < 1.01 * radratio))).all()
    assert np.allclose(np.isfinite(canonly['radj_err1']),
                       np.isfinite(canonly['rade_err1']))
    assert ((~np.isfinite(canonly['rade_err1'])) |
            ((canonly['rade_err1'] / canonly['radj_err1'] > 0.99 * radratio) &
             (canonly['rade_err1'] / canonly['radj_err1'] <
              1.01 * radratio))).all()
    assert np.allclose(np.isfinite(canonly['radj_err2']),
                       np.isfinite(canonly['rade_err2']))
    assert ((~np.isfinite(canonly['rade_err2'])) |
            ((canonly['rade_err2'] / canonly['radj_err2'] > 0.99 * radratio) &
             (canonly['rade_err2'] / canonly['radj_err2'] <
              1.01 * radratio))).all()

    # these flags at least have the right number of good values
    assert not canonly['flag_kepler'].any()
    assert not canonly['flag_k2'].any()
    assert canonly['flag_tran'].all()

    # discovery and confirmation years make sense
    assert (canonly['year_discovered'] >= 2018).all()
    assert not np.isfinite(canonly['year_confirmed']).any()

    # add the K2 candidates to our final composite table
    toiadd = dftoi[cols][newcan].copy()
    comp = pd.concat((comp, toiadd), verify_integrity=True, ignore_index=True)

    ###################
    # FINAL ADDITIONS #
    ###################

    # create the estimate mass/radius columns
    badm = (np.isfinite(comp['masse']) ^ np.isfinite(comp['massj']))
    assert badm.sum() == 0

    badr = (np.isfinite(comp['rade']) ^ np.isfinite(comp['radj']))
    assert badr.sum() == 0

    badr = (np.isfinite(comp['rade_err1']) ^ np.isfinite(comp['radj_err1']))
    assert badr.sum() == 0

    badr = (np.isfinite(comp['rade_err2']) ^ np.isfinite(comp['radj_err2']))
    assert badr.sum() == 0

    getrad = np.isfinite(comp['masse']) & (~np.isfinite(comp['rade']))

    r1 = getrad & (comp['masse'] < 2.04)
    r2 = getrad & (comp['masse'] < 132) & (comp['masse'] >= 2.04)
    r3 = getrad & (comp['masse'] < 26600) & (comp['masse'] >= 132)
    r4 = getrad & (comp['masse'] >= 26600)

    comp.insert(comp.columns.get_loc('rade')+1, 'rade_est', np.nan)
    comp.insert(comp.columns.get_loc('radj')+1, 'radj_est', np.nan)
    comp.loc[r1, 'rade_est'] = 10.**(np.log10(comp.loc[r1, 'masse']) * 0.279 +
                                     0.00346)
    comp.loc[r2, 'rade_est'] = 10.**(np.log10(comp.loc[r2, 'masse']) * 0.589 -
                                     0.0925)
    comp.loc[r3, 'rade_est'] = 10.**(np.log10(comp.loc[r3, 'masse']) * -0.044 +
                                     1.25)
    comp.loc[r4, 'rade_est'] = 10.**(np.log10(comp.loc[r4, 'masse']) * 0.881 -
                                     2.85)

    comp.loc[getrad, 'radj_est'] = comp.loc[getrad, 'rade_est'] / radratio

    getmass = np.isfinite(comp['rade']) & (~np.isfinite(comp['masse']))

    m1 = getmass & (comp['rade'] < 1.23)
    m2 = getmass & (comp['rade'] < 11.1) & (comp['rade'] >= 1.23)
    # note that m3 is degenerate and this is a hack just to make sure
    # everything has values
    m3 = getmass & (comp['rade'] < 14.3) & (comp['rade'] >= 11.1)
    m4 = getmass & (comp['rade'] >= 14.3)

    comp.insert(comp.columns.get_loc('masse')+1, 'masse_est', np.nan)
    comp.insert(comp.columns.get_loc('massj')+1, 'massj_est', np.nan)
    comp.loc[m1, 'masse_est'] = 10.**((np.log10(comp.loc[m1, 'rade']) -
                                       0.00346) / 0.2790)
    comp.loc[m2, 'masse_est'] = 10.**((np.log10(comp.loc[m2, 'rade']) +
                                       0.0925) / 0.589)
    comp.loc[m3, 'masse_est'] = 10.**((np.log10(comp.loc[m3, 'rade']) -
                                       1.25) / -0.044)
    comp.loc[m4, 'masse_est'] = 10.**((np.log10(comp.loc[m4, 'rade']) +
                                       2.85) / 0.881)

    comp.loc[getmass, 'massj_est'] = comp.loc[getmass, 'masse_est'] / massrat

    # save our final version of the data frame to use in making all the plots
    comp.to_csv('data/exoplots_data.csv', index=False)

    return dfcon, dfkoi, dfk2, dftoi, comp


"""
    JavaScript code to create a CSV file from selected data points in a plot

    Parameters
    ----------

"""
csv_creation = """
    function makeCSV(sources, keys, header) {
        const nsources = sources.length;
        const ncolumns = keys.length;
        const lines = [header];
        for (let nn = 0; nn < nsources; nn++) {
            var source = sources[nn];
            const nrows = source.selected.indices.length;
            
            for (let ii = 0; ii < nrows; ii++) {
                let row = [];
                for (let jj = 0; jj < ncolumns; jj++) {
                    const column = keys[jj];
                    var ind = source.selected.indices[ii];
                    row.push(source.data[column][ind].toString());
                }
                lines.push(row.join(', '));
            }
        }
        return lines.join('\\n').concat('\\n');
    };

    // adapted from:
    // https://stackoverflow.com/questions/21012580/is-it-possible-to-write-
    // data-to-file-using-only-javascript
    var textFile = null;
    function makeTextFile(text) {
      var data = new Blob([text], {type: 'text/plain'});

      // If we are replacing a previously generated file we need to
      // manually revoke the object URL to avoid memory leaks.
      if (textFile !== null) {
        window.URL.revokeObjectURL(textFile);
      }

      textFile = window.URL.createObjectURL(data);

      return textFile;
    };

    var link = document.createElement('a');
    // link.setAttribute('download', 'exoplots_download.txt');
    link.setAttribute('target', '_blank');
    link.href = makeTextFile(makeCSV(sources, keys, header));
    document.body.appendChild(link);

    // wait for the link to be added to the document
    window.requestAnimationFrame(function() {
      var event = new MouseEvent('click');
      link.dispatchEvent(event);
      document.body.removeChild(link);
    });
    """

deselect = r"""
const nglyphs = glyphs.length;
var some = 0;
for (let nn = 0; nn < nglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    
    if (!glyph.visible){
        source.selected.indices = [];
    }

    if (source.selected.indices.length == 0){
        glyph.glyph.line_alpha = glyph.nonselection_glyph.line_alpha;
        glyph.glyph.fill_alpha = glyph.nonselection_glyph.fill_alpha;
        glyph.change.emit();
    }
    else {
        glyph.glyph.line_alpha = alphas[nn];
        glyph.glyph.fill_alpha = alphas[nn];
        glyph.change.emit();
        some = 1;
    }
    var newnum = source.selected.indices.length.toLocaleString('en-US');
    newnum = '(' + newnum + ')';
    var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
    legend.label['value'] = newstr;
}
if (some == 0){
    for (let nn = 0; nn < nglyphs; nn++) {
        var glyph = glyphs[nn];
        var source = glyph.data_source;
        var legend = legends[nn];
        
        glyph.glyph.line_alpha = alphas[nn];
        glyph.glyph.fill_alpha = alphas[nn];
        glyph.change.emit();
        
        var newnum = source.data['planet'].length.toLocaleString('en-US');
        newnum = '(' + newnum + ')';
        var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
        legend.label['value'] = newstr;
    }
}
"""

openurl = """
const nsources = sources.length;
const urls = [];
for (let nn = 0; nn < nsources; nn++) {
    var source = sources[nn];
    const nrows = source.selected.indices.length;
    
    for (let ii = 0; ii < nrows; ii++) {
        var ind = source.selected.indices[ii];
        urls.push(source.data['url'][ind]);
        
    }
}
if (urls.length == 1){
    window.open(urls[0]);
}
"""

sldeselect = r"""
const nglyphs = glyphs.length;
var some = 0;
var loop = 0;
while (some == 0 && loop < 3){
    for (let nn = 0; nn < nglyphs; nn++) {
        var glyph = glyphs[nn];
        var source = glyph.data_source;
        var legend = legends[nn];
        
        if (!glyph.visible){
            source.selected.indices = [];
        }
    
        if (source.selected.indices.length == 0){
            glyph.glyph.line_alpha = glyph.nonselection_glyph.line_alpha;
            glyph.glyph.fill_alpha = glyph.nonselection_glyph.fill_alpha;
            glyph.change.emit();
        }
        else {
            glyph.glyph.line_alpha = alphas[nn];
            glyph.glyph.fill_alpha = alphas[nn];
            glyph.change.emit();
            some = 1;
        }
        var newnum = source.selected.indices.length.toLocaleString('en-US');
        newnum = '(' + newnum + ')';
        var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
        legend.label['value'] = newstr;
    }
    if (some == 0){
        var minyr = slider.value[0];
        var maxyr = slider.value[1];
        for (let nn = 0; nn < nglyphs; nn++) {
            var glyph = glyphs[nn];
            var source = glyph.data_source;
            var selected = [];
            for (let ii = 0; ii < source.data['planet'].length; ii++) {
                if (source.data['year'][ii] >= minyr && 
                        source.data['year'][ii] <= maxyr){
                    selected.push(ii);
                }
            }
            if (glyph.visible){
                source.selected.indices = selected;
            }
            else {
                source.selected.indices = [];
            }
            source.change.emit();
        }
    }
    loop = loop + 1;
}
"""

reset = r"""
const nglyphs = glyphs.length;
for (let nn = 0; nn < nglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    
    glyph.glyph.line_alpha = alphas[nn];
    glyph.glyph.fill_alpha = alphas[nn];
    glyph.change.emit();
    
    var newnum = source.data['planet'].length.toLocaleString('en-US');
    newnum = '(' + newnum + ')';
    var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
    legend.label['value'] = newstr;
}
"""

yearselect = """
const mglyphs = glyphs.length;
var minyr = slider.value[0];
var maxyr = slider.value[1];
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var selected = [];
    for (let ii = 0; ii < source.data['planet'].length; ii++) {
        if (source.data['year'][ii] >= minyr && 
                source.data['year'][ii] <= maxyr){
            selected.push(ii);
        }
    }
    if (glyph.visible){
        source.selected.indices = selected;
    }
    else {
        source.selected.indices = [];
    }
    source.change.emit();
}

label.text = minyr + '\u2013' + maxyr;
label.change.emit();

""" + sldeselect

unselect = """
const mglyphs = glyphs.length;
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    source.selected.indices = [];
    source.change.emit();
}
""" + deselect

sliderselect = """
const mglyphs = glyphs.length;
var minyr = slider.value[0];
var maxyr = slider.value[1];
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    
    if (!glyph.visible){
        source.selected.indices = [];
    }
    else {
        var selected = [];
        for (let ii = 0; ii < source.selected.indices.length; ii++) {
            var iyr = source.data['year'][source.selected.indices[ii]];
            if (iyr >= minyr &&  iyr <= maxyr){
                selected.push(source.selected.indices[ii]);
            }
        }
        source.selected.indices = selected;
    }
    source.change.emit();
}

""" + sldeselect

nowvis = """
var tmp = [];
for (var i=0;i<cb_obj.data_source.data['year'].length;i++) {
  tmp.push(i);
}
cb_obj.data_source.selected.indices=tmp;

""" + sliderselect

playpause = """

async function advance(slider, button) {
    var i = 0;
    var mpause = 0;
    while (button.active){
        i = i + 1;
        var vals = slider.value;
        var nextval = vals[1] + 1;
        if (nextval > slider.end){
            if (mpause > 0){
                nextval = vals[0];
                mpause = 0;
            }
            else{
                nextval = nextval - 1;
                mpause = mpause + 1;
            }
        }
        slider.value = [vals[0], nextval];
        await new Promise(r => setTimeout(r, 1000));
    }
}

async function startloop(slider, button) {
    const result = await advance(slider, button);    
}

if (cb_obj.active){
    cb_obj.label = "\u2759\u2759 Pause";
    startloop(slider, cb_obj);
}
else {
    cb_obj.label = "\u25b6 Play";
}
"""

singleslide = r"""
var nglyphs = glyphs.length;
var some = 0;
var loop = 0;
while (some == 0 && loop < 3){
    for (let nn = 0; nn < nglyphs; nn++) {
        var glyph = glyphs[nn];
        var source = glyph.data_source;
        var legend = legends[nn];
        let whisker_s = whisker_sources[nn].data;
        let radius_s = radius_sources[nn].data;
        
        // Reset the whisker_source to empty
        for (var ii in whisker_s) {
            whisker_s[ii] = [];
        }
        for (var ii in radius_s) {
            radius_s[ii] = [];
        }
        if (!glyph.visible){
            source.selected.indices = [];
        }

        if (source.selected.indices.length == 0){
            glyph.glyph.line_alpha = glyph.nonselection_glyph.line_alpha;
            glyph.glyph.fill_alpha = glyph.nonselection_glyph.fill_alpha;
            glyph.change.emit();
        }
        else {
            glyph.glyph.line_alpha = alphas[nn];
            glyph.glyph.fill_alpha = alphas[nn];
            glyph.change.emit();
            some = 1;
            for (var ii of source.selected.indices) {
                // Iterate through all columns of the data and push to 
                // whisker source
                for (var jj in whisker_s) {
                        whisker_s[jj].push(source.data[jj][ii]);
                }
                for (var jj in radius_s) {
                        radius_s[jj].push(source.data[jj][ii]);
                }
            }
        }
        var newnum = source.selected.indices.length.toLocaleString('en-US');
        newnum = '(' + newnum + ')';
        var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
        legend.label['value'] = newstr;
        whisker_sources[nn].change.emit();
        radius_sources[nn].change.emit();
    }
    if (some == 0){
        var maxerror = slider.value;
        for (let nn = 0; nn < nglyphs; nn++) {
            var glyph = glyphs[nn];
            var source = glyph.data_source;
            var selected = [];
            let whisker_s = whisker_sources[nn].data;
            let radius_s = radius_sources[nn].data;
            // Reset the whisker_source to empty
            for (var ii in whisker_s) {
                whisker_s[ii] = [];
            }
            for (var ii in radius_s) {
                radius_s[ii] = [];
            }
            
            for (let ii = 0; ii < source.data['planet'].length; ii++) {
                if ((source.data['masserror'][ii] <= maxerror && 
                        source.data['masserror'][ii] > 0) || maxerror == 100){
                    selected.push(ii);
                }
            }
            if (glyph.visible){
                source.selected.indices = selected;
                for (var ii of source.selected.indices) {
                    // Iterate through all columns of the data and push to 
                    // whisker source
                    for (var jj in whisker_s) {
                            whisker_s[jj].push(source.data[jj][ii]);
                    }
                    for (var jj in radius_s) {
                            radius_s[jj].push(source.data[jj][ii]);
                    }
                }
            }
            else {
                source.selected.indices = [];
            }
            source.change.emit();
            whisker_sources[nn].change.emit();
            radius_sources[nn].change.emit();
        }
    }
    loop = loop + 1;
}
"""

massselect = """
const mglyphs = glyphs.length;
var maxerror = slider.value;
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    let whisker_s = whisker_sources[nn].data;
    let radius_s = radius_sources[nn].data;
    var selected = [];
    
    // Reset the whisker_source to empty
    for (var ii in whisker_s) {
        whisker_s[ii] = [];
    }
    for (var ii in radius_s) {
        radius_s[ii] = [];
    }
    for (let ii = 0; ii < source.data['planet'].length; ii++) {
        if ((source.data['masserror'][ii] <= maxerror && 
                source.data['masserror'][ii] > 0) || maxerror == 100){
            selected.push(ii);
            // Iterate through all columns of the data and push to 
            // whisker source
            for (var jj in whisker_s) {
                    whisker_s[jj].push(source.data[jj][ii]);
            }
            for (var jj in radius_s) {
                    radius_s[jj].push(source.data[jj][ii]);
            }
        }
    }
    if (glyph.visible){
        source.selected.indices = selected;
    }
    else {
        source.selected.indices = [];
    }
    source.change.emit();
    whisker_sources[nn].change.emit();
    radius_sources[nn].change.emit();
}
""" + singleslide

sing_sliderselect = """
const mglyphs = glyphs.length;
var maxerror = slider.value;
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    let whisker_s = whisker_sources[nn].data;
    let radius_s = radius_sources[nn].data;
    
    // Reset the whisker_source to empty
    for (var ii in whisker_s) {
        whisker_s[ii] = [];
    }
    for (var ii in radius_s) {
        radius_s[ii] = [];
    }
    
    if (!glyph.visible){
        source.selected.indices = [];
    }
    else {
        var selected = [];
        for (let ii = 0; ii < source.selected.indices.length; ii++) {
            var iyr = source.data['masserror'][source.selected.indices[ii]];
            if ((iyr <= maxerror && iyr > 0) || maxerror == 100){
                selected.push(source.selected.indices[ii]);
                // Iterate through all columns of the data and push to 
                // whisker source
                for (var jj in whisker_s) {
                        whisker_s[jj].push(source.data[jj][ii]);
                }
                for (var jj in radius_s) {
                        radius_s[jj].push(source.data[jj][ii]);
                }
            }
        }
        source.selected.indices = selected;
    }
    source.change.emit();
    whisker_sources[nn].change.emit();
    radius_sources[nn].change.emit();
}

""" + singleslide
