import matplotlib.pyplot as plt
import numpy as np
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import sunpy.map
import sunpy.sun
from sunpy.coordinates import get_body_heliographic_stonyhurst
from sunpy.net import Fido
from sunpy.net import attrs as a

######################################################################

stereo = (a.Instrument('EUVI') &
          a.Time('2011-11-01', '2011-11-01T00:10:00'))

aia = (a.Instrument('AIA') &
       a.vso.Sample(24 * u.hour) &
       a.Time('2011-11-01', '2011-11-02'))

wave = a.Wavelength(19.5 * u.nm, 19.5 * u.nm)

res = Fido.search(wave, aia | stereo)

files = Fido.fetch(res)

######################################################################

maps = sunpy.map.Map(sorted(files))

######################################################################

maps = [m.resample((1024, 1024)*u.pix) for m in maps]

######################################################################

maps[0].meta['rsun_ref'] = sunpy.sun.constants.radius.to_value(u.m)

######################################################################

shape_out = (180, 360)  # This is set deliberately low to reduce memory consumption

header = sunpy.map.make_fitswcs_header(shape_out,
                                       SkyCoord(0, 0, unit=u.deg,
                                                frame="heliographic_stonyhurst",
                                                obstime=maps[0].date),
                                       scale=[180 / shape_out[0],
                                              360 / shape_out[1]] * u.deg / u.pix,
                                       wavelength=int(maps[0].meta['wavelnth']) * u.AA,
                                       projection_code="CAR")
out_wcs = WCS(header)

######################################################################

coordinates = tuple(map(sunpy.map.all_coordinates_from_map, maps))

######################################################################

weights = [coord.transform_to("heliocentric").z.value for coord in coordinates]

######################################################################

weights = [(w / np.nanmax(w)) ** 3 for w in weights]
for w in weights:
    w[np.isnan(w)] = 0

######################################################################

array, _ = reproject_and_coadd(maps, out_wcs, shape_out,
                               input_weights=weights,
                               reproject_function=reproject_interp,
                               match_background=True,
                               background_reference=0)

######################################################################

outmap = sunpy.map.Map((array, header))
outmap.plot_settings = maps[0].plot_settings
outmap.nickname = 'AIA + EUVI/A + EUVI/B'

outfig = plt.figure(figsize=(10, 5))
ax = plt.subplot(projection=out_wcs)
im = outmap.plot(vmin=400)

lon, lat = ax.coords

lon.set_coord_type("longitude")
lon.coord_wrap = 180
lon.set_format_unit(u.deg)
lat.set_coord_type("latitude")
lat.set_format_unit(u.deg)

lon.set_axislabel('Heliographic Longitude', minpad=0.8)
lat.set_axislabel('Heliographic Latitude', minpad=0.9)
lon.set_ticks(spacing=25*u.deg, color='k')
lat.set_ticks(spacing=15*u.deg, color='k')

plt.colorbar(im, ax=ax)

# Reset the view to pixel centers
_ = ax.axis((0, shape_out[1], 0, shape_out[0]))

outfig.savefig("./images/repro_comp.png", transparent=True)
