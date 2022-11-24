# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:16:39 2022

@author: yinlu
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import sys
from basic_units import cm
from matplotlib import patches

from matplotlib.pyplot    import *
from mpl_toolkits.basemap import pyproj
from matplotlib           import rc

#==================================================================================================================
    
class Basemap(Basemap):
    
    def ellipse(self, x0, y0, a, b, n, ax=None, **kwargs):
        
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.

        For a description of the properties of ellipsis, please refer to [1].

        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].

        Extra keyword ``ax`` can be used to override the default axis instance.

        Other \**kwargs passed on to matplotlib.patches.Polygon

        RETURNS
            poly : a maptplotlib.patches.Polygon object.

        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse
        """
        
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b

        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = [self(x0+a, y0)]
        AZ = np.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if np.allclose(0., y0) and (np.allclose(90., az) or
                np.allclose(270., az)):
                continue

            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * np.pi / 360. * (az + 90.)
            A = dist[0] * np.sin(azr)
            B = dist[1] * np.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)

            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))

        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)

        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)

        return poly

#==================================================================================================================

"""
Options: --gal, --eq, --ham, --transp
"""

param_list = sys.argv[1:]
file_names = []
for item in param_list:
    if item not in ['--eq','--gal','--ham']:
        file_names.append(item)
    else:
        break

projection_list = ['hammer', 'moll']
if '--ham' in sys.argv:
    projection = projection_list[0]
else:
    projection = projection_list[1]

p_size  = 2 # default rcParams['lines.markersize'] ** 2 which is 36
if '--transp' in sys.argv:
    p_alpha = 0.013 # transparency of the point to mimic a density map
else:
    p_alpha = 1.0
 
# Coordinates
# -----------
cmbdipole_l   = 263.99 # CMB dibole
cmbdipole_b   = 48.26
cmbdipole_c   = '#e41a1c' # color to use

cmboctopole_l   = 239 
cmbdctopole_b   = 64.3
cmboctopole_c  ='#e41a1c'

cmbqua_l   = 224.2 
cmbqua_b   = 69.2
cmbqua_c  ='#e41a1c'

Planckva_l   = 212 
Planckva_b   = -13
Planckva_c  ='#e41a1c'

WMAP9VA_l   = 219 
WMAP9VA_b  = -24
WMAP9VA_c  ='#e41a1c'

PlanckDM_l   = 227 
PlanckDM_b   = -15
PlanckDM_c  ='#e41a1c'

WMAP5DM_l   = 224
WMAP5DM_b   = -22
WMAP5DM_c  ='#e41a1c'

PlanckPA_l   = 218
PlanckPA_b   = -21
PlanckPA_c  ='#e41a1c'

WMAP9PA_l   = 227 
WMAP9PA_b   = -27
WMAP9PA_c  ='#e41a1c'

alpha_l = 330.13 # Fine structure constant dipole (http://arxiv.org/abs/1202.4758)
alpha_b = -13.16
alpha_c = '#4daf4a'

dipoles =  {'CMB octopole': [cmboctopole_l, cmbdctopole_b, cmboctopole_c],
            'CMB quadrupole': [cmbqua_l, cmbqua_b, cmbqua_c],}
dipoles2 =  {'CMB dipole': [cmbdipole_l,cmbdipole_b,cmbdipole_c],}

CMBHPA =  { 'Planck-VA': [Planckva_l, Planckva_b, Planckva_c],
            'Planck-DM': [PlanckDM_l, PlanckDM_b, PlanckDM_c],
            'Planck-PA': [PlanckPA_l, PlanckPA_b, PlanckPA_c],
            }

wmap =  {   'WMAP9-VA': [WMAP9VA_l, WMAP9VA_b, WMAP9VA_c],
            'WNAP5-DM': [WMAP5DM_l, WMAP5DM_b, WMAP5DM_c],
            'WMAP9-PA': [WMAP9PA_l, WMAP9PA_b, WMAP9PA_c],
            }

ra_patches, dec_patches, l_patches, b_patches = [], [], [], []
for file_name in file_names:
    if '--eq' in sys.argv: # coordinates in file_name are (RA,DEC)
        ra_list, dec_list = np.loadtxt(file_name,unpack=True)
        gal_coord = SkyCoord(ra_list,dec_list,frame='fk5',unit='deg').galactic
        l_list = gal_coord.l.deg # longitude
        b_list = gal_coord.b.deg # latitude
    elif '--gal' in sys.argv: # coordinates in file_name are (l,b)
        l_list, b_list = np.loadtxt(file_name,unpack=True)
        eq_coord = SkyCoord(l_list,b_list,frame='galactic',unit='deg').fk5
        ra_list  = eq_coord.ra.deg
        dec_list  = eq_coord.dec.deg
    ra_patches.append(ra_list)
    dec_patches.append(dec_list)
    l_patches.append(l_list)
    b_patches.append(b_list)
      
fig = plt.figure(figsize=(18,10))

ax2 = plt.subplot(122)
m = Basemap(projection=projection,lat_0=0,lon_0=360) #180)
m.drawparallels(np.arange(-90,90,30))
m.drawmeridians(np.arange(-180,180,60)) #(0,360,30))


# Markers:
for key in dipoles.keys():
    x, y = m(dipoles[key][0],dipoles[key][1]) # direction
    m.scatter(x,y,s=30,marker='x',color=dipoles[key][2],lw=2.5)
    x, y = m(dipoles[key][0]-2,dipoles[key][1]+3)
    ax2.text(x,y,'{}'.format(key),color='black',fontsize=7,va='top',ha='right')
    x, y = m(dipoles[key][0]-180.0,-1.0*dipoles[key][1]) # anti-direction
    m.scatter(x,y,s=30,marker='x',color=dipoles[key][2],lw=2.5)
    
for key in dipoles2.keys():
    x, y = m(dipoles2[key][0],dipoles2[key][1]) # direction
    m.scatter(x,y,s=30,marker='x',color=dipoles2[key][2],lw=2.5)
    x, y = m(dipoles2[key][0]+4,dipoles2[key][1]+2)
    ax2.text(x,y,'{}'.format(key),color='black',fontsize=7,va='top',ha='right')
    x, y = m(dipoles2[key][0]-180.0,-1.0*dipoles2[key][1]) # anti-direction
    m.scatter(x,y,s=30,marker='x',color=dipoles2[key][2],lw=2.5)
    
for key in CMBHPA.keys():
    x, y = m(CMBHPA[key][0],CMBHPA[key][1]) # direction
    m.scatter(x,y,s=10,marker='>',color=CMBHPA[key][2],lw=2.5)
    x, y = m(CMBHPA[key][0]-24,CMBHPA[key][1]+1)
    ax2.text(x,y,'{}'.format(key),color='black',fontsize=5,va='top',ha='right')
    x, y = m(CMBHPA[key][0]-180.0,-1.0*CMBHPA[key][1]) # anti-direction
    m.scatter(x,y,s=10,marker='>',color=CMBHPA[key][2],lw=2.5)
    #mark  marker=(5, 0)  (5, 1) (5,2)
    
for key in wmap.keys():
    x, y = m(wmap[key][0],wmap[key][1]) # direction
    m.scatter(x,y,s=10,marker='>',color='black',lw=2.5)
    x, y = m(wmap[key][0]+5,wmap[key][1]+1)
    ax2.text(x,y,'{}'.format(key),color='black',fontsize=5,va='top',ha='right')
    x, y = m(wmap[key][0]-180.0,-1.0*wmap[key][1]) # anti-direction
    m.scatter(x,y,s=10,marker='>',color='black',lw=2.5)




x1, y1=(307,9)
width, height=(15, 15)
xval,yval = m(x1,y1)
m.scatter(xval,yval,marker='>',color='saddlebrown',lw=2.0)
#m.ellipse(x1,y1, width, height,100,facecolor='blue',alpha=0.5,linewidth=0,zorder=1)  # only needed for ellipse contour
#xval,yval = m(x1-180.0, -1.0*y1)
#m.scatter(xval,yval,color='orange',edgecolor='none',s=20,zorder=2)
#m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='blue',alpha=0.5,linewidth=0,zorder=1)
x, y = m(x1+45.0, y1+2.4)
ax2.text(x,y,'Great Attractor',color='black',fontsize=7,va='top',ha='left')

x1, y1=(282,6)
width, height=(11, 6)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='firebrick',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='firebrick',alpha=0.6,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='firebrick',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='firebrick',alpha=0.6,linewidth=0,zorder=1)
x, y = m(x1+10.0, y1+6.0)
ax2.text(x,y,'Bulk flow',color='black',fontsize=7,va='top',ha='left')

x1, y1=(290,30)
width, height=(20, 15)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='darkviolet',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='darkviolet',alpha=0.55,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='darkviolet',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='darkviolet',alpha=0.55,linewidth=0,zorder=1)
x, y = m(x1+8.0, y1-3.0)
ax2.text(x,y,'Dark flow',color='black',fontsize=7,va='top',ha='left')



x1, y1=(310.6,-13)
width, height=(18.2, 11.1)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='dodgerblue',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='dodgerblue',alpha=0.7,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='dodgerblue',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='dodgerblue',alpha=0.7,linewidth=0,zorder=1)
x, y = m(x1+10.0, y1-3.0)
ax2.text(x,y,'SNe Ia',color='black',fontsize=7,va='top',ha='left')




x1, y1=(248,44)
width, height=(12.5,8)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='darkorange',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='darkorange',alpha=0.7,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='darkorange',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='darkorange',alpha=0.7,linewidth=0,zorder=1)
x, y = m(x1+5.0, y1-3.0)
ax2.text(x,y,'NVSS',color='black',fontsize=7,va='top',ha='left')

x1, y1=(247,52)
width, height=(14.6,8)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='lightskyblue',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='lightskyblue',alpha=0.6,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='lightskyblue',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='lightskyblue',alpha=0.6,linewidth=0,zorder=1)
x, y = m(x1+15.0, y1+6.0)
ax2.text(x,y,'TGSS',color='black',fontsize=7,va='top',ha='left')


x1, y1=(280,42)
#width, height=(18.2, 11.1)
xval,yval = m(x1,y1)
#m.scatter(x,y,s=30,marker='x',color=dipoles2[key][2],lw=2.5)
m.scatter(xval,yval,marker='x',color='red',lw=2.5)
#m.ellipse(x1,y1, width, height,100,facecolor='grey',alpha=0.5,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,marker='x',color='red',lw=2.5)
#m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='grey',alpha=0.5,linewidth=0,zorder=1)
x, y = m(x1+77.0, y1+2.0)
ax2.text(x,y,'CMB kinematic dipole',color='black',fontsize=7,va='top',ha='left')

#---point need to make sure--------------------
x1, y1=(280,-15)
width, height=(35,20)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='gold',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='yellow',alpha=0.5,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='gold',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='yellow',alpha=0.5,linewidth=0,zorder=1)
x, y = m(x1+19.0, y1-7.0)
ax2.text(x,y,'Galaxy cluster',color='black',fontsize=7,va='top',ha='left')

x1, y1=(238.2,28.8)
width, height=(9,9)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='green',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='green',alpha=0.5,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='green',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='green',alpha=0.7,linewidth=0,zorder=1)
x, y = m(x1+10.0, y1-3.0)
ax2.text(x,y,'CatWISE dipole',color='black',fontsize=7,va='top',ha='left')

x1, y1=(48.8, -5.6)
width, height=(17.2,14.3)
xval,yval = m(x1,y1)
m.scatter(xval,yval,color='grey',edgecolor='none',s=20,zorder=2)
m.ellipse(x1,y1, width, height,100,facecolor='grey',alpha=0.7,linewidth=0,zorder=1)  # only needed for ellipse contour
xval,yval = m(x1-180.0, -1.0*y1)
m.scatter(xval,yval,color='grey',edgecolor='none',s=20,zorder=2)
m.ellipse(x1-180.0, -1.0*y1,width, height,100,facecolor='grey',alpha=0.7,linewidth=0,zorder=1)
x, y = m(x1+19.0, y1-5.0)
ax2.text(x,y,'dipole in the',color='black',fontsize=7,va='top',ha='left')
x, y = m(x1+25.0, y1-9.0)
ax2.text(x,y,'cosmological parameters',color='black',fontsize=7,va='top',ha='left')


#####0819
x1, y1=(264,-17)
xval,yval = m(x1,y1)
m.scatter(xval,yval,marker='+',color='black',lw=2.0)
x, y = m(x1+10.0, y1+8.4)
ax2.text(x,y,'$max(S_-)$',color='black',fontsize=7,va='top',ha='left')

x1, y1=(260,48)
xval,yval = m(x1,y1)
m.scatter(xval,yval,marker='+',color='black',lw=2.0)
x, y = m(x1-7.0, y1+3)
ax2.text(x,y,'$max(S_+)$',color='black',fontsize=7,va='top',ha='left')

# Pole labels on the sphere
x,y = m(-2,0)
ax2.text(x,y,r'(0,0)',color='black',fontsize=9,va='bottom',ha='left')
#x,y = m(188,3)
#ax2.text(x,y,r'(180,0)',color='black',fontsize=9,va='center',ha='right')
x,y = m(0,-90)
ax2.text(x,y,r'(0,-90)',color='black',fontsize=9,va='top',ha='center')
x,y = m(0,90)
ax2.text(x,y,r'(0,90)',color='black',fontsize=9,va='bottom',ha='center')
x,y = m(178,3)
ax2.text(x,y,r'(180,0)',color='black',fontsize=9,va='center',ha='left')
# Sub-labels on the axes:
for b in (-60,-30,30,60):
    x,y = m(180,b)
    if b<0:
        ax2.text(x,y,'$%d^{\circ}$'%b,color='black',fontsize=10,va='top',ha='left')
    if b>0:
        ax2.text(x,y,r'$%d^{\circ}$'%b,color='black',fontsize=10,va='bottom',ha='left')
    for l in (60,120,240,300):
        x,y = m(l-1,0)
        ax2.text(x,y,r"$%d^{\circ}$"%l,color='black',fontsize=8,va='bottom',ha='left')

#plt.tight_layout()
if '--save' in sys.argv:
    fname = 'example/sdss_example1.png'
    plt.savefig(fname,bbox_inches='tight', pad_inches=0)  
plt.show()
