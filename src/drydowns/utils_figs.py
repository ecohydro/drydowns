
import os
import re
import numpy as np
import getpass

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns



# SETTINGS

mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
font_files = mpl.font_manager.findSystemFonts(
    fontpaths=[os.path.join('home', getpass.getuser(), 'Fonts')]
)

for font_file in font_files:
    mpl.font_manager.fontManager.addfont(font_file)



mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
# mpl.rcParams['font.sans-serif'] = 'Source Sans Pro'
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['axes.titlesize'] = 12.0

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


var_dict = {
    'SWC' : {
        'symbol' : r"$\theta$",
        'label' : r"$\theta$ (m$^3$ m$^{-3}$)",
        'units' : r"m$^3$ m$^{-3}$",
    },
    'theta' : {
        'symbol' : r"$\theta$",
        'label' : r"$\theta$ (m$^3$ m$^{-3}$)",
        'units' : r"m$^3$ m$^{-3}$",
    },
    'dtheta' : {
        'symbol' : r"$\frac{d\theta}{dt}$",
        'label' : r"$\frac{d\theta}{dt}$ (m$^3$ m$^{-3}$ day$^{-1}$)",
        'units' : r"m$^3$ m$^{-3}$ day$^{-1}$",
    },
    'dtheta_mm' : {
        'symbol' : r"$\frac{d\theta}{dt}$",
        'label' : r"$\frac{d\theta}{dt}$ (mm day$^{-1}$)",
        'units' : r"mm day$^{-1}$",
    },
    't' : {
        'symbol' : r"$t$",
        'label' : r"$t$ (days)",
        'units' : r"days",
    },
    'et' : {
        'symbol' : r"$ET$",
        'label' : r"$ET$ (mm day$^{-1}$)",
        'units' : r"mm day$^{-1}$",
    },
}



def plot(
        ax, df, x, y, hue=None, label=None, xlim=None, ylim=None,
        kwargs={'s' : 12, 'linewidth' : 0, 'alpha' : 0.8}, legend=True
    ):
    # if hue:
    sns.scatterplot(
        x=x, y=y, hue=hue, data=df, ax=ax, #label=label,
        **kwargs
    )
    # else:
    #     ax.plot(
    #         # self.df.t, self.df['theta'+suff], '.-', label=labels.get(suff), 
    #         df[x], df[y], '.', label=label,
    #         **kwargs #color='k', alpha=0.8
    #     )
    if 'dtheta' in y:
        ykey =  'dtheta_mm' if re.search(re.compile(r'theta_mm'), y) else 'dtheta'
    else:
        ykey = y.rsplit('_', 1)[0]
    ax.set_xlabel(
        # var_dict.get(re.match(r'^([a-zA-Z]+(?:_mm)?)(?:_obs)?$', x).group(1))['label']
        var_dict.get(x.rsplit('_', 1)[0])['label']
    )
    ax.set_ylabel(
        # var_dict.get(re.match(r'^([a-zA-Z]+(?:_mm)?)(?:_obs)?$', y).group(1))['label']
        # var_dict.get(y.rsplit('_', 1)[0])['label']
        var_dict.get(ykey)['label']
    )
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 1))

    return ax

def plot_theta(
        ax, df, hue=None, xlim=None, ylim=None, 
        legend=True, kwargs={'s' : 12, 'linewidth' : 0, 'alpha' : 0.8},
    ):
    plot(
        ax=ax, df=df, x='t', y='theta', hue=hue, label='Observed', 
        xlim=xlim, ylim=ylim, legend=legend,kwargs=kwargs
    )

    return ax

def plot_dtheta(
        ax, df, hue=None, units='mm', xlim=None, ylim=None,
        legend=True, kwargs={'s' : 12, 'linewidth' : 0, 'alpha' : 0.8}, plot_et=False,
    ):
    # if plot_et and 'et' in self.df.columns:
    #     plot(
    #         ax=ax, df=self.df, x='theta_obs', y='et', label='ET', xlim=xlim, 
    #         ylim=ylim, legend=legend, kwargs={'color' : 'r', 'alpha' : 0.8}
    #     )
    y = 'dtheta'
    if units == 'mm':
        y += '_mm'
    plot(
        ax=ax, df=df, x='theta', y=y, hue=hue, label='Observed', 
        xlim=xlim, ylim=ylim, legend=legend, kwargs=kwargs
    )
    return ax

def plot_drydown(
        df, axs=None, hue=None, units='mm', xlim=None, ylim=None,
        kwargs={'s' : 12, 'linewidth' : 0, 'alpha' : 0.8}, plot_et=True,
    ):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    plot_theta(axs[0], df, xlim=xlim, ylim=ylim, hue=hue, legend=False, kwargs=kwargs)
    plot_dtheta(
        axs[1], df, units=units, xlim=xlim, ylim=ylim, hue=hue, legend=True, kwargs=kwargs, plot_et=plot_et
    )
    axs[1].legend(loc='upper left', bbox_to_anchor=(0.01, 1))
    # lat long with 2 digits
    # plt.gcf().suptitle(
    #     f"{self.id[0]}, {self.info['IGBP']}, z = {self.z} m ({self.info['latitude']:.2f}, {self.info['longitude']:.2f})"
    # )
    plt.tight_layout()
    return axs