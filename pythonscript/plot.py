import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm 
import matplotlib.ticker as ticker
from matplotlib.patches import Circle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d
#%% 3D 
tb = pd.read_table('..//results//mini_MSR_dfdf_3D_v1.table', header=7)
tb_arr = tb.to_numpy()
tb_arr = tb_arr[::4]
# tb_arr = np.concatenate((tb_arr[::4], tb_arr[-1]))
ax = plt.figure().add_subplot(projection='3d')

for i in range(len(tb_arr)):
    lst_of_val = tb_arr[i].item().split()
    x = float(lst_of_val[1])
    y = float(lst_of_val[2])
    z = float(lst_of_val[0])
    Bx = float(lst_of_val[3])
    By = float(lst_of_val[4])
    Bz = float(lst_of_val[5])
    
    norm = np.sqrt(Bx**2+By**2+Bz**2)
    
    ax.quiver(x, y, z, Bx, By, Bz, length=norm*300, normalize=True, cmap=plt.cm.jet)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=180, azim=0, roll=0)
# plt.show()

#%% contour
contour_data =  pd.read_table('..//results//mini_MSR_dfdf_centerplane_v1.table',delim_whitespace=True, header=8)
contour_data.columns = ['x', 'z', 'Bx', 'By', 'Bz', 'B']

B = contour_data.pivot_table(index='x', columns='z', values='B').T.values

X_unique = np.sort(contour_data.x.unique())
Z_unique = np.sort(contour_data.z.unique())
X, Z = np.meshgrid(X_unique, Z_unique)

fig, ax = plt.subplots(figsize=(6,6))

levels = []
for i in [-8, -7, -6, -5]:
    for j in range(1, 10):
        levels.append(j*10**i)


ctf = ax.contourf(X, Z, B, levels[7:],  locator=ticker.LogLocator())


ax.set_xlabel('X-axis')
ax.set_ylabel('Z-axis')
cbar = plt.colorbar(ctf, ticks=levels[::4], label='B [T]')
cbar.ax.set_yticklabels(['{:0.0e}'.format(i) for i in levels[::4]])


plt.tight_layout()

#%% 3D contour
fig = plt.figure(figsize=(5,7))
ax = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212)
contour_data =  pd.read_table('..//results//mini_MSR_dfdf_XZ_centerplane_v1.table',delim_whitespace=True, header=8)
contour_data.columns = ['x', 'z', 'Bx', 'By', 'Bz', 'B']

B = contour_data.pivot_table(index='x', columns='z', values='B').T.values

X_unique = np.sort(contour_data.x.unique())
Z_unique = np.sort(contour_data.z.unique())
X, Z = np.meshgrid(X_unique, Z_unique)


levels = []
for i in [-8, -7, -6, -5]:
    for j in range(1, 10):
        levels.append(j*10**i)


ctf = ax.contourf(X, Z, B, levels[7:],  locator=ticker.LogLocator(), offset=0)


# draw cube 
inner_sl = 0.636/2
r = [-inner_sl, inner_sl]
X, Y = np.meshgrid(r, r)
one = np.full(4, inner_sl).reshape(2, 2)
ax.plot_wireframe(X,Y,one, alpha=0.5)
ax.plot_wireframe(X,Y,-one, alpha=0.5)
ax.plot_wireframe(X,-one,Y, alpha=0.5)
ax.plot_wireframe(X,one,Y, alpha=0.5)
ax.plot_wireframe(one,X,Y, alpha=0.5)
ax.plot_wireframe(-one,X,Y, alpha=0.5)

outer_sl = 0.72/2
r = [-outer_sl, outer_sl]
X, Y = np.meshgrid(r, r)
one = np.full(4, outer_sl).reshape(2, 2)
ax.plot_wireframe(X,Y,one, alpha=0.5)
ax.plot_wireframe(X,Y,-one, alpha=0.5)
ax.plot_wireframe(X,-one,Y, alpha=0.5)
ax.plot_wireframe(X,one,Y, alpha=0.5)
ax.plot_wireframe(one,X,Y, alpha=0.5)
ax.plot_wireframe(-one,X,Y, alpha=0.5)

p = Circle((0, 0), 0.03)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=outer_sl, zdir="y")

ax.set_xlabel('X-axis')
ax.set_ylabel('Z-axis')
ax.set_zlabel('Y-axis')

B = contour_data.pivot_table(index='x', columns='z', values='B').T.values

X_unique = np.sort(contour_data.x.unique())
Z_unique = np.sort(contour_data.z.unique())
X, Z = np.meshgrid(X_unique, Z_unique)
ctf2 = ax2.contourf(X, Z, B, levels[7:],  locator=ticker.LogLocator())


ax2.set_xlabel('X-axis')
ax2.set_ylabel('Z-axis')
cbar = plt.colorbar(ctf2, ticks=levels[::4], label='B [T]')
cbar.ax.set_yticklabels(['{:0.0e}'.format(i) for i in levels[::4]])
ax2.set_ylim(ax2.get_ylim()[::-1])
plt.tight_layout()
#%% Bz along z

meas = pd.read_csv('..//mapping_data//mini_MSR_20240604.csv', comment='#')
#dfdf
dfdf = pd.read_table('..//results//mini_MSR_dfdf_v1.table', header=4)
dfdf = dfdf.to_numpy()
#ffff
ffff = pd.read_table('..//results//mini_MSR_ffff_v1.table', header=4)
ffff = ffff.to_numpy()
#dfff
dfff = pd.read_table('..//results//mini_MSR_dfff_v1.table', header=4)
dfff = dfff.to_numpy()

#dddd
dddd = pd.read_table('..//results//mini_MSR_dddd_v1.table', header=4)
dddd = dddd.to_numpy()


fig, ax = plt.subplots(figsize=(8,4))
# inset on flat region
x1, x2, y1, y2 = 0.05, 0.3, -0.4, 0.55  # subregion of the original image
axins = ax.inset_axes(
    [0.18, 0.15, 0.4, 0.4],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[],)

# inset near front panel
x1, x2, y1, y2 = 0.3, 0.35, -0.4, 6  # subregion of the original image
axins1 = ax.inset_axes(
    [0.55, 0.65, 0.3, 0.3],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[],)

ax.errorbar(0.01*meas['z '], meas['avg'], meas['std'], fmt='*--' ,label='Measurement')
axins.errorbar(0.01*meas['z '], meas['avg'], meas['std'], fmt='*--' ,label='Measurement')
axins1.errorbar(0.01*meas['z '], meas['avg'], meas['std'], fmt='*--' ,label='Measurement')


labels = ['dfdf',
          'ffff',
          'dfff',
          'dddd']

ct = 0
for arangement in [dfdf, ]: #, ffff, dfff, dddd
    z = []
    Bz = []
    for i in range(len(arangement)):
        lst_of_val = arangement[i].item().split()
        z.append(float(lst_of_val[0]))
        Bz.append(float(lst_of_val[1]) * 10**6)
        
    ax.plot(z, Bz, '-', label='Simulation {}'.format(labels[ct]))
    axins.plot(z, Bz, '-')
    axins1.plot(z, Bz, '-')
    ct += 1
# highlight shieldings
# outer
ax.axvspan(0.372, 0.373, alpha=0.7, color='red', label='Shielding Walls')
ax.axvspan(-0.373, -0.372, alpha=0.7, color='red')

# inner
ax.axvspan(0.3265, 0.3275, alpha=0.7, color='red')
ax.axvspan(-0.3275, -0.3265, alpha=0.7, color='red')

# ax.grid()
axins.grid()
axins1.grid()


ax.indicate_inset_zoom(axins, edgecolor="black")
ax.indicate_inset_zoom(axins1, edgecolor="black")

ax.set_xlabel('z [m]')
ax.set_ylabel(r'B [$\mu$T]')
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])
ax.set_position([box.x0 + box.width * 0.1, box.y0,
                  box.width * 0.9, box.height])

# Put a legend below current axis
ax.legend(loc='center left', bbox_to_anchor=(-0.5, 0),
          fancybox=True, shadow=True, nrow=1)

fig.show()


# fig = plt.figure(figsize=(7,3.5))

# gs = fig.add_gridspec(4,6)
# ax = fig.add_subplot(gs[:, :4])
# ax2 = fig.add_subplot(gs[:2, 4:])
# ax1 = fig.add_subplot(gs[2:, 4:])

# ax.errorbar(0.01*meas['z '], meas['avg'], meas['std'], fmt='*--' ,label='Measurement')
# ax1.errorbar(0.01*meas['z '], meas['avg'], meas['std'], fmt='*--' ,label='Measurement')
# ax2.errorbar(0.01*meas['z '], meas['avg'], meas['std'], fmt='*--' ,label='Measurement')
# labels = ['dfdf',
#          'ffff',
#          'dfff',
#          'dddd']

# ct = 0
# for arangement in [dfdf, ]: #, ffff, dfff, dddd
#     z = []
#     Bz = []
#     for i in range(len(arangement)):
#         lst_of_val = arangement[i].item().split()
#         z.append(float(lst_of_val[0]))
#         Bz.append(float(lst_of_val[1]) * 10**6)
        
#     ax.plot(z, Bz, '-', label='Simulation {}'.format(labels[ct]))
#     ax1.plot(z, Bz, '-', color='C1')
#     ax2.plot(z, Bz, '-', color='C1')
#     ct += 1
# # highlight shieldings
# # outer
# ax.axvspan(0.372, 0.373, alpha=0.7, color='red', label='Shielding Walls')
# ax.axvspan(-0.373, -0.372, alpha=0.7, color='red')

# # inner
# ax.axvspan(0.3265, 0.3275, alpha=0.7, color='red')
# ax.axvspan(-0.3275, -0.3265, alpha=0.7, color='red')

# ax1.set_xlim(0.05, 0.3)
# ax1.set_ylim( -0.4, 0.5)

# ax2.set_xlim(0.3, 0.35)
# ax2.set_ylim(-0.4, 6)

# ax.grid()
# ax1.grid()
# ax2.grid()

# ax.set_xlabel('z [m]')
# ax.set_ylabel(r'B [$\mu$T]')
# # Shrink current axis's height by 10% on the bottom
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0 + box.height * 0.1,
# #                  box.width, box.height * 0.9])

# # # Put a legend below current axis
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
# #           fancybox=True, shadow=True, ncol=5)
# ax.legend()
# plt.tight_layout()
# fig.show()










